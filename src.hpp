#pragma once
#include "simulator.hpp"
namespace sjtu {

// Attention per round:
//   S = Q * K^T   (r x r),   softmax rows,   O = P * V   (r x 512)
//
// Cost / memory model (from simulator.hpp):
//   * MatMul(A[m x n], B[n x q]) costs 5 * size(A) * size(B) = 5 m n^2 q,
//     quadratic in the inner dimension n => splitting the 512-dim inner
//     product into small chunks is much cheaper than one big MatMul.
//   * Score = 1e6 * acc * min(1.25, 1.5e10 / cycles) * exp(-m / 1.5e6),
//     m = peak SRAM usage (elements).  We have a huge cycle headroom, so
//     we optimize almost exclusively for low peak SRAM.
//
// Memory strategy (peak SRAM only a few thousand elements):
//   * K^T (dim x r) and V (r x dim) stacks live in HBM and are updated
//     with HBM concat/transpose instructions (25x cost, cycles are cheap).
//   * S = Q K^T is computed in W-column chunks: Q[:, c0:c0+W] (r x W) and
//     K^T[c0:c0+W, :] (W x r) are extracted in HBM (GetColumn / GetRow +
//     Concat, all in HBM), moved to SRAM, multiplied and accumulated.
//     Only the two small chunks plus the r x r accumulator live in SRAM.
//   * PV phase: one output row at a time, O[a,:] = sum_j P[a,j] * V[j,:],
//     fetching each V row (1 x 512) into SRAM only when needed.
//   * The answer is assembled in HBM.  Two Run() calls per round make the
//     KT/V parking moves race-free (no queued calc uses them in Run #2).

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t n_rounds = keys.size();
  const size_t dim = 512;
  const size_t W = 16; // inner-dimension chunk width for QK^T

  auto alloc = [&](const std::string &name) {
    return matrix_memory_allocator.Allocate(name);
  };

  Matrix *KT = nullptr; // dim x r, in HBM
  Matrix *V = nullptr;  // r x dim, always stays in HBM

  for (size_t round = 0; round < n_rounds; ++round) {
    const size_t r = round + 1;
    Matrix *Q = rater.GetNextQuery(); // r x dim, in HBM

    // ==================== Run #1: S = Q K^T, E = exp(S) =============
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

    // ---- chunked accumulation: S += Q[:, c0:c0+W] * KT[c0:c0+W, :] ----
    Matrix *S = nullptr; // r x r, SRAM
    for (size_t c0 = 0; c0 < dim; c0 += W) {
      // build QC = Q[:, c0:c0+W] (r x W) in HBM
      Matrix *QC = nullptr;
      for (size_t c = c0; c < c0 + W; ++c) {
        Matrix *qcol = alloc("qcol");
        gpu_sim.GetColumn(Q, c, qcol, kInGpuHbm); // r x 1, HBM
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
      gpu_sim.MoveMatrixToSharedMem(QC); // IO, r x W
      gpu_sim.MoveMatrixToSharedMem(KC); // IO, W x r
      Matrix *term = alloc("term");
      gpu_sim.MatMul(QC, KC, term); // r x r, cost 5 r W^2 r
      if (S == nullptr) {
        S = term;
      } else {
        Matrix *nS = alloc("S");
        gpu_sim.MatAdd(S, term, nS); // cost r^2
        gpu_sim.ReleaseMatrix(S);
        gpu_sim.ReleaseMatrix(term);
        S = nS;
      }
      gpu_sim.ReleaseMatrix(QC);
      gpu_sim.ReleaseMatrix(KC);
    }
    gpu_sim.ReleaseMatrix(Q);

    Matrix *E = alloc("E");
    gpu_sim.MatExp(S, E); // r x r, cost 30 r^2
    gpu_sim.ReleaseMatrix(S);

    gpu_sim.Run(false, &matrix_memory_allocator);

    // ==================== Run #2: O = softmax(E) * V ================
    // KT and V both stay in HBM; fetch each V row into SRAM when needed.

    Matrix *ANS = nullptr; // assembled in HBM
    for (size_t a = 0; a < r; ++a) {
      Matrix *erow = alloc("erow");
      gpu_sim.GetRow(E, a, erow, kInSharedMemory); // 1 x r
      Matrix *rowsum = alloc("rowsum");
      gpu_sim.Sum(erow, rowsum); // 1 x 1
      Matrix *prow = alloc("prow");
      gpu_sim.MatDiv(erow, rowsum, prow); // 1 x r (softmax row)
      gpu_sim.ReleaseMatrix(erow);
      gpu_sim.ReleaseMatrix(rowsum);

      // O[a, :] = sum_j P[a, j] * V[j, :]
      Matrix *orow = nullptr; // 1 x dim, SRAM
      for (size_t j = 0; j < r; ++j) {
        Matrix *vrow = alloc("vrow");
        gpu_sim.GetRow(V, j, vrow, kInGpuHbm); // 1 x dim, HBM
        gpu_sim.MoveMatrixToSharedMem(vrow);   // IO
        Matrix *pij = alloc("pij");
        gpu_sim.GetColumn(prow, j, pij, kInSharedMemory); // 1 x 1
        Matrix *term = alloc("term");
        gpu_sim.MatMul(pij, vrow, term); // 1 x dim, cost 5 dim
        if (orow == nullptr) {
          orow = term;
        } else {
          Matrix *nrow = alloc("orow");
          gpu_sim.MatAdd(orow, term, nrow); // cost dim
          gpu_sim.ReleaseMatrix(orow);
          gpu_sim.ReleaseMatrix(term);
          orow = nrow;
        }
        gpu_sim.ReleaseMatrix(pij);
        gpu_sim.ReleaseMatrix(vrow);
      }
      gpu_sim.ReleaseMatrix(prow);

      gpu_sim.MoveMatrixToGpuHbm(orow); // IO
      if (ANS == nullptr) {
        ANS = orow;
      } else {
        Matrix *nANS = alloc("ANS");
        gpu_sim.Concat(ANS, orow, nANS, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(ANS);
        gpu_sim.ReleaseMatrix(orow);
        ANS = nANS;
      }
    }
    gpu_sim.ReleaseMatrix(E);

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
