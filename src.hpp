#pragma once
#include "simulator.hpp"
#include <algorithm>
namespace sjtu {

// Attention per round:
//   S = Q * K^T   (r x r),   softmax rows,   O = P * V   (r x 512)
//
// Cost / memory model (from simulator.hpp):
//   * MatMul(A[m x n], B[n x q]) costs 5 * size(A) * size(B) = 5 m n^2 q,
//     quadratic in the inner dimension n => splitting the 512-dim inner
//     product into small chunks is far cheaper than one big MatMul.
//   * Score = 1e6 * acc * min(1.25, 1.5e10 / cycles) * exp(-m / 1.5e6),
//     m = peak SRAM usage (elements).  We have a huge cycle headroom
//     (cycles used ~3e9 << 1.2e10 needed for the 1.25 cap), so we optimize
//     almost exclusively for low peak SRAM.
//
// Memory strategy (peak SRAM ~ 1000 elements):
//   * K^T (dim x r) and V (r x dim) stacks live in HBM and are updated
//     with HBM concat/transpose instructions.
//   * Q is split into row blocks of BLK rows (in HBM).  S_b = Q_b K^T is
//     accumulated over W-wide inner-dimension chunks: Q_b[:, c] (b x W)
//     and KT[c, :] (W x r) are extracted in HBM, moved into SRAM, and
//     multiplied.  Each block's accumulator S_b (b x r) is parked in HBM
//     between chunks, so at most one S_b plus the two small chunks live
//     in SRAM at any time.
//   * PV phase: per block, compute the softmax rows P[a, :] from E_b,
//     then O[a, :] = P[a, :] V is evaluated via 16-wide column chunks of
//     V (fetched into SRAM one at a time); output rows are assembled in
//     HBM.  Two Run() calls per round keep the parking moves race-free.

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t n_rounds = keys.size();
  const size_t dim = 512;
  const size_t BLK = 8;  // query rows per block
  const size_t W = 8;    // inner-dim chunk width for QK^T
  const size_t WV = 16;  // column chunk width of V for the PV product

  auto alloc = [&](const std::string &name) {
    return matrix_memory_allocator.Allocate(name);
  };

  Matrix *KT = nullptr; // dim x r, in HBM
  Matrix *V = nullptr;  // r x dim, in HBM

  for (size_t round = 0; round < n_rounds; ++round) {
    const size_t r = round + 1;
    Matrix *Q = rater.GetNextQuery(); // r x dim, in HBM
    const size_t nb = (r + BLK - 1) / BLK;

    // ---- update KT in HBM: append keys[round]^T as a new column ----
    gpu_sim.Transpose(keys[round], kInGpuHbm); // in-place, now dim x 1
    if (KT == nullptr) {
      Matrix *kt0 = alloc("KT");
      gpu_sim.Copy(keys[round], kt0, kInGpuHbm);
      KT = kt0;
    } else {
      Matrix *nKT = alloc("KT");
      gpu_sim.Concat(KT, keys[round], nKT, 1, kInGpuHbm);
      gpu_sim.ReleaseMatrix(KT);
      KT = nKT;
    }
    // ---- update V in HBM: append values[round] as a new row ----
    if (V == nullptr) {
      Matrix *v0 = alloc("V");
      gpu_sim.Copy(values[round], v0, kInGpuHbm);
      V = v0;
    } else {
      Matrix *nV = alloc("V");
      gpu_sim.Concat(V, values[round], nV, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(V);
      V = nV;
    }

    // ---- split Q into row blocks, entirely in HBM ----
    std::vector<Matrix *> qblocks;
    for (size_t start = 0; start < r; start += BLK) {
      const size_t b = std::min(BLK, r - start);
      Matrix *blk = nullptr;
      for (size_t a = 0; a < b; ++a) {
        Matrix *qrow = alloc("qrow");
        gpu_sim.GetRow(Q, start + a, qrow, kInGpuHbm); // 1 x dim, HBM
        if (blk == nullptr) {
          blk = qrow;
        } else {
          Matrix *nblk = alloc("qb");
          gpu_sim.Concat(blk, qrow, nblk, 0, kInGpuHbm);
          gpu_sim.ReleaseMatrix(blk);
          gpu_sim.ReleaseMatrix(qrow);
          blk = nblk;
        }
      }
      qblocks.push_back(blk);
    }
    gpu_sim.ReleaseMatrix(Q);

    // ==================== Run #1: S_b = Q_b K^T, E_b = exp(S_b) =====
    std::vector<Matrix *> S(nb, nullptr); // block accumulators, parked in HBM
    for (size_t c0 = 0; c0 < dim; c0 += W) {
      // build KC = KT[c0:c0+W, :] (W x r) in HBM
      Matrix *KC = nullptr;
      for (size_t c = c0; c < c0 + W; ++c) {
        Matrix *krow = alloc("krow");
        gpu_sim.GetRow(KT, c, krow, kInGpuHbm); // 1 x r, HBM
        if (KC == nullptr) {
          KC = krow;
        } else {
          Matrix *nKC = alloc("KC");
          gpu_sim.Concat(KC, krow, nKC, 0, kInGpuHbm);
          gpu_sim.ReleaseMatrix(KC);
          gpu_sim.ReleaseMatrix(krow);
          KC = nKC;
        }
      }
      gpu_sim.MoveMatrixToSharedMem(KC); // IO
      for (size_t bidx = 0; bidx < nb; ++bidx) {
        // build QC = Q_b[:, c0:c0+W] (b x W) in HBM
        Matrix *QB = qblocks[bidx];
        Matrix *QC = nullptr;
        for (size_t c = c0; c < c0 + W; ++c) {
          Matrix *qcol = alloc("qcol");
          gpu_sim.GetColumn(QB, c, qcol, kInGpuHbm); // b x 1, HBM
          if (QC == nullptr) {
            QC = qcol;
          } else {
            Matrix *nQC = alloc("QC");
            gpu_sim.Concat(QC, qcol, nQC, 1, kInGpuHbm);
            gpu_sim.ReleaseMatrix(QC);
            gpu_sim.ReleaseMatrix(qcol);
            QC = nQC;
          }
        }
        gpu_sim.MoveMatrixToSharedMem(QC);   // IO
        if (S[bidx] != nullptr) {
          gpu_sim.MoveMatrixToSharedMem(S[bidx]); // IO, fetch accumulator
        }
        Matrix *term = alloc("term");
        gpu_sim.MatMul(QC, KC, term); // b x r, cost 5 (b W)(W r)
        gpu_sim.ReleaseMatrix(QC);
        if (S[bidx] == nullptr) {
          S[bidx] = term;
        } else {
          Matrix *nS = alloc("S");
          gpu_sim.MatAdd(S[bidx], term, nS); // cost b r
          gpu_sim.ReleaseMatrix(S[bidx]);
          gpu_sim.ReleaseMatrix(term);
          S[bidx] = nS;
        }
        gpu_sim.MoveMatrixToGpuHbm(S[bidx]); // park accumulator (IO)
      }
      gpu_sim.ReleaseMatrix(KC);
    }
    // E_b = exp(S_b); park E_b in HBM
    std::vector<Matrix *> E(nb, nullptr);
    for (size_t bidx = 0; bidx < nb; ++bidx) {
      gpu_sim.MoveMatrixToSharedMem(S[bidx]); // IO
      E[bidx] = alloc("E");
      gpu_sim.MatExp(S[bidx], E[bidx]); // b x r
      gpu_sim.ReleaseMatrix(S[bidx]);
      gpu_sim.MoveMatrixToGpuHbm(E[bidx]); // park (IO)
    }

    gpu_sim.Run(false, &matrix_memory_allocator);

    // ==================== Run #2: O = softmax(E) * V ================
    // build V column chunks (r x WV) once, in HBM
    std::vector<Matrix *> vchunks;
    for (size_t c0 = 0; c0 < dim; c0 += WV) {
      Matrix *VC = nullptr;
      for (size_t c = c0; c < c0 + WV; ++c) {
        Matrix *vcol = alloc("vcol");
        gpu_sim.GetColumn(V, c, vcol, kInGpuHbm); // r x 1, HBM
        if (VC == nullptr) {
          VC = vcol;
        } else {
          Matrix *nVC = alloc("VC");
          gpu_sim.Concat(VC, vcol, nVC, 1, kInGpuHbm);
          gpu_sim.ReleaseMatrix(VC);
          gpu_sim.ReleaseMatrix(vcol);
          VC = nVC;
        }
      }
      vchunks.push_back(VC);
    }

    Matrix *ANS = nullptr; // assembled in HBM
    for (size_t bidx = 0; bidx < nb; ++bidx) {
      const size_t b = std::min(BLK, r - bidx * BLK);
      gpu_sim.MoveMatrixToSharedMem(E[bidx]); // IO, fetch E_b
      // softmax rows of this block
      std::vector<Matrix *> prows;
      for (size_t a = 0; a < b; ++a) {
        Matrix *erow = alloc("erow");
        gpu_sim.GetRow(E[bidx], a, erow, kInSharedMemory); // 1 x r
        Matrix *rowsum = alloc("rowsum");
        gpu_sim.Sum(erow, rowsum); // 1 x 1
        Matrix *prow = alloc("prow");
        gpu_sim.MatDiv(erow, rowsum, prow); // 1 x r (softmax row)
        gpu_sim.ReleaseMatrix(erow);
        gpu_sim.ReleaseMatrix(rowsum);
        prows.push_back(prow);
      }
      gpu_sim.ReleaseMatrix(E[bidx]);
      // O[a, :] = P[a, :] (1 x r) * V (r x dim), via V column chunks
      std::vector<Matrix *> orows(b, nullptr);
      for (size_t cidx = 0; cidx < vchunks.size(); ++cidx) {
        Matrix *VC = vchunks[cidx];
        gpu_sim.MoveMatrixToSharedMem(VC); // IO
        for (size_t a = 0; a < b; ++a) {
          Matrix *ochunk = alloc("ochunk");
          gpu_sim.MatMul(prows[a], VC, ochunk); // 1 x WV, cost 5 r (r WV)
          gpu_sim.MoveMatrixToGpuHbm(ochunk);   // IO
          if (orows[a] == nullptr) {
            orows[a] = ochunk;
          } else {
            Matrix *nrow = alloc("orow");
            gpu_sim.Concat(orows[a], ochunk, nrow, 1, kInGpuHbm);
            gpu_sim.ReleaseMatrix(orows[a]);
            gpu_sim.ReleaseMatrix(ochunk);
            orows[a] = nrow;
          }
        }
        if (bidx + 1 < nb) {
          gpu_sim.MoveMatrixToGpuHbm(VC); // park for the next block
        } else {
          gpu_sim.ReleaseMatrix(VC);
        }
      }
      for (size_t a = 0; a < b; ++a) {
        gpu_sim.ReleaseMatrix(prows[a]);
        // append O[a, :] (1 x dim, HBM) to the answer
        if (ANS == nullptr) {
          ANS = orows[a];
        } else {
          Matrix *nANS = alloc("ANS");
          gpu_sim.Concat(ANS, orows[a], nANS, 0, kInGpuHbm);
          gpu_sim.ReleaseMatrix(ANS);
          gpu_sim.ReleaseMatrix(orows[a]);
          ANS = nANS;
        }
      }
    }

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ANS); // ANS is in HBM; released by the rater
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
