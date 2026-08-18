#pragma once
#include "simulator.hpp"
namespace sjtu {

// Attention per round:
//   S = Q * K^T          (r x r),   softmax rows,  O = P * V  (r x 512)
//
// Cost model notes (from simulator.hpp):
//   MatMul(A[m x n], B[n x q]) costs 5 * (m n) * (n q) = 5 m n^2 q,
//   which is quadratic in the inner dimension n.  Therefore a rank-1
//   decomposition (n = 1 pieces) is ~n times cheaper than one big matmul.
//   Elementwise row ops (GetRow/Sum/MatDiv/MulNum) are linear in size.
// We also release every temporary as early as possible to keep the peak
// SRAM usage low (the score decays exponentially with peak SRAM).

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t n_rounds = keys.size();
  const size_t dim = 512;

  auto alloc = [&](const std::string &name) {
    return matrix_memory_allocator.Allocate(name);
  };

  // Move all keys / values to SRAM once; they persist across rounds.
  for (size_t i = 0; i < n_rounds; ++i) {
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);
  }

  Matrix *KT = nullptr; // stacked transposed keys: dim x r, in SRAM
  Matrix *V = nullptr;  // stacked values:         r x dim, in SRAM

  for (size_t round = 0; round < n_rounds; ++round) {
    const size_t r = round + 1;
    Matrix *Q = rater.GetNextQuery(); // r x dim, in HBM
    gpu_sim.MoveMatrixToSharedMem(Q);

    // ---- update K^T stack (append keys[round]^T as a new column) ----
    Matrix *kt_col = alloc("kt_col");
    gpu_sim.Copy(keys[round], kt_col, kInSharedMemory); // 1 x dim in SRAM
    gpu_sim.Transpose(kt_col, kInSharedMemory);         // dim x 1
    if (KT == nullptr) {
      KT = kt_col;
    } else {
      Matrix *new_KT = alloc("KT");
      gpu_sim.Concat(KT, kt_col, new_KT, 1, kInSharedMemory);
      gpu_sim.ReleaseMatrix(KT);
      gpu_sim.ReleaseMatrix(kt_col);
      KT = new_KT;
    }
    gpu_sim.ReleaseMatrix(keys[round]); // data now lives inside KT

    // ---- update V stack (append values[round] as a new row) ----
    if (V == nullptr) {
      Matrix *v0 = alloc("V");
      gpu_sim.Copy(values[round], v0, kInSharedMemory);
      V = v0;
    } else {
      Matrix *new_V = alloc("V");
      gpu_sim.Concat(V, values[round], new_V, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V);
      V = new_V;
    }
    gpu_sim.ReleaseMatrix(values[round]); // data now lives inside V

    // ---- S = Q K^T via rank-1 accumulation over the 512 columns ----
    // S = sum_c Q[:, c] (r x 1) * KT[c, :] (1 x r)
    Matrix *S = nullptr;
    for (size_t c = 0; c < dim; ++c) {
      Matrix *qc = alloc("qc");
      gpu_sim.GetColumn(Q, c, qc, kInSharedMemory); // r x 1, cost r
      Matrix *kc = alloc("kc");
      gpu_sim.GetRow(KT, c, kc, kInSharedMemory); // 1 x r, cost r
      Matrix *term = alloc("term");
      gpu_sim.MatMul(qc, kc, term); // r x r, cost 5 r^2
      if (S == nullptr) {
        S = term;
      } else {
        Matrix *new_S = alloc("S");
        gpu_sim.MatAdd(S, term, new_S); // cost r^2
        gpu_sim.ReleaseMatrix(S);
        gpu_sim.ReleaseMatrix(term);
        S = new_S;
      }
      gpu_sim.ReleaseMatrix(qc);
      gpu_sim.ReleaseMatrix(kc);
    }
    gpu_sim.ReleaseMatrix(Q);

    // ---- E = exp(S) ----
    Matrix *E = alloc("E");
    gpu_sim.MatExp(S, E); // cost 30 r^2
    gpu_sim.ReleaseMatrix(S);

    // ---- row-wise: P[a, :] = E[a, :] / sum(E[a, :]);
    //      O[a, :] = P[a, :] * V = sum_j P[a, j] * V[j, :]
    //      assemble O in HBM to keep SRAM usage small ----
    Matrix *ANS = nullptr; // assembled in HBM
    for (size_t a = 0; a < r; ++a) {
      Matrix *erow = alloc("erow");
      gpu_sim.GetRow(E, a, erow, kInSharedMemory); // 1 x r, cost r
      Matrix *rowsum = alloc("rowsum");
      gpu_sim.Sum(erow, rowsum); // 1 x 1, cost r
      Matrix *prow = alloc("prow");
      gpu_sim.MatDiv(erow, rowsum, prow); // 1 x r, cost 16 r
      gpu_sim.ReleaseMatrix(erow);
      gpu_sim.ReleaseMatrix(rowsum);

      // O[a, :] = P[a, :] (1 x r) * V (r x dim), one MatMul per row.
      Matrix *orow = alloc("orow");
      gpu_sim.MatMul(prow, V, orow); // cost 5 * r * r * dim
      gpu_sim.ReleaseMatrix(prow);

      // move this output row to HBM and assemble the answer there
      gpu_sim.MoveMatrixToGpuHbm(orow); // cost 300 dim
      if (ANS == nullptr) {
        ANS = orow;
      } else {
        Matrix *new_ANS = alloc("ANS");
        gpu_sim.Concat(ANS, orow, new_ANS, 0, kInGpuHbm); // 25 * size
        gpu_sim.ReleaseMatrix(ANS);
        gpu_sim.ReleaseMatrix(orow);
        ANS = new_ANS;
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
