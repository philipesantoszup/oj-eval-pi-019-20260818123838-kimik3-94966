#pragma once
#include "simulator.hpp"
namespace sjtu {

// Attention per round:
//   S = Q * K^T   (r x r),   softmax rows,   O = P * V   (r x 512)
//
// Cost / memory model (from simulator.hpp):
//   * MatMul(A[m x n], B[n x q]) costs 5 * size(A) * size(B) = 5 m n^2 q,
//     quadratic in the inner dimension => rank-1 accumulation (n = 1
//     pieces) is ~512x cheaper for QK^T than one big MatMul.
//   * Score = 1e6 * acc * min(1.25, 1.5e10 / cycles) * exp(-m / 1.5e6),
//     with m = peak SRAM usage (in elements).  Our cycle count is far
//     below the 1.25 cap, so we optimize almost exclusively for SRAM.
//
// Memory strategy:
//   * K^T (dim x r) and V (r x dim) stacks live in HBM between rounds and
//     are updated with HBM concat/transpose instructions (25x cost, but
//     cycles are not the bottleneck).
//   * Q is split into small row blocks entirely in HBM (GetRow + Concat).
//   * Run #1 (S phase): bring KT and one Q block at a time into SRAM,
//     accumulate S_b = Q_b K^T by rank-1 updates, keep only E_b = exp(S_b)
//     (total r^2 elements) in SRAM.
//   * Run #2 (PV phase): park KT back to HBM, bring V into SRAM, compute
//     each output row as MatMul(softmax row, V), assemble the answer in
//     HBM.  Splitting into two Run() calls guarantees no IO instruction
//     can race ahead of the calc instructions that use the same matrix.

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t n_rounds = keys.size();
  const size_t dim = 512;
  const size_t BLK = 8; // query rows per block

  auto alloc = [&](const std::string &name) {
    return matrix_memory_allocator.Allocate(name);
  };

  Matrix *KT = nullptr; // dim x r, in HBM between rounds
  Matrix *V = nullptr;  // r x dim, in HBM between rounds

  for (size_t round = 0; round < n_rounds; ++round) {
    const size_t r = round + 1;
    Matrix *Q = rater.GetNextQuery(); // r x dim, in HBM

    // ==================== Run #1: preparation + S phase ============
    if (V != nullptr) {
      gpu_sim.MoveMatrixToGpuHbm(V); // park V (it was left in SRAM by PV)
    }
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
    const size_t n_blocks = (r + BLK - 1) / BLK;
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

    // ---- S phase: for each block, S_b = sum_c Q_b[:,c] * KT[c,:] ----
    gpu_sim.MoveMatrixToSharedMem(KT); // dim x r -> SRAM
    std::vector<Matrix *> eblocks;
    for (size_t bidx = 0; bidx < n_blocks; ++bidx) {
      Matrix *QB = qblocks[bidx];
      gpu_sim.MoveMatrixToSharedMem(QB); // b x dim -> SRAM
      Matrix *S = nullptr;
      for (size_t c = 0; c < dim; ++c) {
        Matrix *qc = alloc("qc");
        gpu_sim.GetColumn(QB, c, qc, kInSharedMemory); // b x 1, cost b
        Matrix *kc = alloc("kc");
        gpu_sim.GetRow(KT, c, kc, kInSharedMemory); // 1 x r, cost r
        Matrix *term = alloc("term");
        gpu_sim.MatMul(qc, kc, term); // b x r, cost 5 b r
        if (S == nullptr) {
          S = term;
        } else {
          Matrix *nS = alloc("S");
          gpu_sim.MatAdd(S, term, nS); // cost b r
          gpu_sim.ReleaseMatrix(S);
          gpu_sim.ReleaseMatrix(term);
          S = nS;
        }
        gpu_sim.ReleaseMatrix(qc);
        gpu_sim.ReleaseMatrix(kc);
      }
      gpu_sim.ReleaseMatrix(QB);
      Matrix *E = alloc("E");
      gpu_sim.MatExp(S, E); // b x r, cost 30 b r
      gpu_sim.ReleaseMatrix(S);
      eblocks.push_back(E); // keep in SRAM (total r^2 elements)
    }

    gpu_sim.Run(false, &matrix_memory_allocator);

    // ==================== Run #2: PV phase ==========================
    // No instruction in this batch touches KT, so parking it is safe.
    gpu_sim.MoveMatrixToGpuHbm(KT);
    gpu_sim.MoveMatrixToSharedMem(V); // r x dim -> SRAM

    Matrix *ANS = nullptr; // assembled in HBM
    size_t row_index = 0;
    for (size_t bidx = 0; bidx < n_blocks; ++bidx) {
      Matrix *E = eblocks[bidx]; // b x r, SRAM
      const size_t b = std::min(BLK, r - bidx * BLK);
      for (size_t a = 0; a < b; ++a, ++row_index) {
        Matrix *erow = alloc("erow");
        gpu_sim.GetRow(E, a, erow, kInSharedMemory); // 1 x r
        Matrix *rowsum = alloc("rowsum");
        gpu_sim.Sum(erow, rowsum); // 1 x 1
        Matrix *prow = alloc("prow");
        gpu_sim.MatDiv(erow, rowsum, prow); // 1 x r (softmax row)
        gpu_sim.ReleaseMatrix(erow);
        gpu_sim.ReleaseMatrix(rowsum);
        // O[row, :] = P[row, :] (1 x r) * V (r x dim)
        Matrix *orow = alloc("orow");
        gpu_sim.MatMul(prow, V, orow); // 1 x dim, cost 5 r r dim
        gpu_sim.ReleaseMatrix(prow);
        // assemble the answer in HBM
        gpu_sim.MoveMatrixToGpuHbm(orow);
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
    }

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ANS); // ANS is in HBM; released by the rater
    (void)row_index;
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
