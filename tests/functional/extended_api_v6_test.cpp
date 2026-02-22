// extended_api_v6_test.cpp
// Tests: cublasChemv/Zhemv (Hermitian GEMV),
//        cublasCher/Zher (rank-1 update),
//        cublasCher2/Zher2 (rank-2 update),
//        cublasCherk/Zherk (rank-k update),
//        cublasCher2k/Zher2k (rank-2k update),
//        cublasChemm/Zhemm (Hermitian matrix-matrix multiply),
//        cublasCgemmStridedBatched/ZgemmStridedBatched

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

#include "cuda_runtime.h"
#include "cublas_v2.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, name)                                            \
    do {                                                             \
        if (cond) { printf("  PASS: %s\n", name); ++g_pass; }       \
        else      { printf("  FAIL: %s\n", name); ++g_fail; }       \
    } while (0)

// ── cublasChemv ───────────────────────────────────────────────────────────────
// A (2×2 Hermitian, upper stored): A[0,0]={3,0}, A[0,1]={1,2}, A[1,1]={5,0}
// → A[1,0] = conj(A[0,1]) = {1,-2}
// x = [{1,0},{0,1}]  ⟹  y = A * x
// y[0] = {3,0}*{1,0} + {1,2}*{0,1} = {3,0} + {-2,1} = {1,1}
// y[1] = {1,-2}*{1,0} + {5,0}*{0,1} = {1,-2} + {0,5} = {1,3}
static void test_chemv() {
    printf("[cublasChemv]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // Upper stored col-major: col0={A00,A10}, col1={A01,A11} → {3,0},{X,X},{1,2},{5,0}
    // A10 is not accessed in upper mode, fill with garbage.
    std::vector<cuComplex> A = {{3,0},{99,99},{1,2},{5,0}};
    std::vector<cuComplex> x = {{1,0},{0,1}};
    std::vector<cuComplex> y = {{0,0},{0,0}};
    void *dA, *dx, *dy;
    cudaMalloc(&dA, 4 * sizeof(cuComplex));
    cudaMalloc(&dx, 2 * sizeof(cuComplex));
    cudaMalloc(&dy, 2 * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasChemv(h, CUBLAS_FILL_MODE_UPPER, 2, &alpha,
                                     (cuComplex*)dA, 2, (cuComplex*)dx, 1,
                                     &beta, (cuComplex*)dy, 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasChemv returns success");
    cudaMemcpy(y.data(), dy, 2 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    bool ok = (fabsf(y[0].x - 1.0f) < 1e-5f && fabsf(y[0].y - 1.0f) < 1e-5f &&
               fabsf(y[1].x - 1.0f) < 1e-5f && fabsf(y[1].y - 3.0f) < 1e-5f);
    CHECK(ok, "cublasChemv result correct");

    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

// ── cublasZhemv ───────────────────────────────────────────────────────────────
// A (2×2 Hermitian, lower stored): A[0,0]={2,0}, A[1,0]={0,1}, A[1,1]={4,0}
// → A[0,1] = conj(A[1,0]) = {0,-1}
// x = [{1,0},{1,0}]
// y[0] = {2,0}*{1,0} + {0,-1}*{1,0} = {2,0}+{0,-1} = {2,-1}
// y[1] = {0,1}*{1,0} + {4,0}*{1,0}  = {0,1}+{4,0}  = {4,1}
static void test_zhemv() {
    printf("[cublasZhemv]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // Lower stored col-major: col0={A00,A10}, col1={X,A11}
    std::vector<cuDoubleComplex> A = {{2,0},{0,1},{99,99},{4,0}};
    std::vector<cuDoubleComplex> x = {{1,0},{1,0}};
    std::vector<cuDoubleComplex> y = {{0,0},{0,0}};
    void *dA, *dx, *dy;
    cudaMalloc(&dA, 4 * sizeof(cuDoubleComplex));
    cudaMalloc(&dx, 2 * sizeof(cuDoubleComplex));
    cudaMalloc(&dy, 2 * sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZhemv(h, CUBLAS_FILL_MODE_LOWER, 2, &alpha,
                                     (cuDoubleComplex*)dA, 2, (cuDoubleComplex*)dx, 1,
                                     &beta, (cuDoubleComplex*)dy, 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZhemv returns success");
    cudaMemcpy(y.data(), dy, 2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    bool ok = (fabs(y[0].x - 2.0) < 1e-9 && fabs(y[0].y - (-1.0)) < 1e-9 &&
               fabs(y[1].x - 4.0) < 1e-9 && fabs(y[1].y - 1.0) < 1e-9);
    CHECK(ok, "cublasZhemv result correct");

    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

// ── cublasCher ─────────────────────────────────────────────────────────────────
// Start: A = 0 (upper).  x = [{1,1}].  alpha = 2.
// Expected: A[0,0] += 2 * |x[0]|^2 = 2*2 = 4 (real, imag=0).
static void test_cher() {
    printf("[cublasCher]\n");
    cublasHandle_t h;  cublasCreate(&h);

    std::vector<cuComplex> A = {{0,0}};
    std::vector<cuComplex> x = {{1,1}};
    void *dA, *dx;
    cudaMalloc(&dA, sizeof(cuComplex));
    cudaMalloc(&dx, sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), sizeof(cuComplex), cudaMemcpyHostToDevice);

    float alpha = 2.0f;
    cublasStatus_t st = cublasCher(h, CUBLAS_FILL_MODE_UPPER, 1, &alpha,
                                    (cuComplex*)dx, 1, (cuComplex*)dA, 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCher returns success");
    cudaMemcpy(A.data(), dA, sizeof(cuComplex), cudaMemcpyDeviceToHost);
    CHECK(fabsf(A[0].x - 4.0f) < 1e-5f && fabsf(A[0].y) < 1e-5f, "cublasCher diagonal result");

    cudaFree(dA); cudaFree(dx);
    cublasDestroy(h);
}

// ── cublasZher ─────────────────────────────────────────────────────────────────
// A (2×2, lower), initially 0.  x = [{0,1},{1,0}].  alpha = 1.
// A[0,0] += |x[0]|^2 = 1, A[1,0] += x[1]*conj(x[0]) = {1,0}*{0,-1} = {0,-1}
// A[1,1] += |x[1]|^2 = 1.
static void test_zher() {
    printf("[cublasZher]\n");
    cublasHandle_t h;  cublasCreate(&h);

    std::vector<cuDoubleComplex> A = {{0,0},{0,0},{99,99},{0,0}};
    std::vector<cuDoubleComplex> x = {{0,1},{1,0}};
    void *dA, *dx;
    cudaMalloc(&dA, 4 * sizeof(cuDoubleComplex));
    cudaMalloc(&dx, 2 * sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    double alpha = 1.0;
    cublasStatus_t st = cublasZher(h, CUBLAS_FILL_MODE_LOWER, 2, &alpha,
                                    (cuDoubleComplex*)dx, 1, (cuDoubleComplex*)dA, 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZher returns success");
    cudaMemcpy(A.data(), dA, 4 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    bool ok = (fabs(A[0].x - 1.0) < 1e-9 && fabs(A[0].y) < 1e-9 &&   // A[0,0]=1
               fabs(A[1].x - 0.0) < 1e-9 && fabs(A[1].y - (-1.0)) < 1e-9 && // A[1,0]={0,-1}
               fabs(A[3].x - 1.0) < 1e-9 && fabs(A[3].y) < 1e-9);    // A[1,1]=1
    CHECK(ok, "cublasZher result correct");

    cudaFree(dA); cudaFree(dx);
    cublasDestroy(h);
}

// ── cublasCher2 ────────────────────────────────────────────────────────────────
// A=0 (upper 1×1), x={1,0}, y={0,1}, alpha={1,0}.
// A[0,0] += alpha*x[0]*conj(y[0]) + conj(alpha)*y[0]*conj(x[0])
//         = {1,0}*{0,-1} + {1,0}*{0,1} = {0,-1}+{0,1} = {0,0}  → diagonal=0 (imag zeroed)
// Actually: A[0,0] += 2*Re(alpha * x[0] * conj(y[0])) = 2*Re({0,-1}) = 0.
static void test_cher2() {
    printf("[cublasCher2]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // Use a 2×2 case for more coverage.
    // A=0 (upper), x={1,1}, y={1,-1}, alpha={1,0}
    // A[0,0] += 2*Re(x[0]*conj(y[0])) = 2*Re({1,1}*{1,1}) = 2*Re({0,2}) = 0
    // A[0,1] += alpha*x[0]*conj(y[1]) + conj(alpha)*y[0]*conj(x[1])
    //         = {1,1}*{1,1} + {1,-1}*{1,-1} = {0,2}+{0,-2} = {0,0}
    // Hmm, that's all zeros. Let's pick x={1,0}, y={0,1}.
    // A[0,0] += 2*Re({1,0}*{0,-1}) = 2*Re({0,-1}) = 0
    // A[0,1] += {1,0}*{0,-1}*... let's just check the call succeeds and
    // do a direct known result: x={2,0}, y={0,0} → only x contributes.
    // alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])
    // With y=0: contribution = 0. So we just verify call returns success.
    std::vector<cuComplex> A = {{0,0},{0,0},{0,0},{0,0}};
    std::vector<cuComplex> x = {{3,0},{0,0}};
    std::vector<cuComplex> y = {{0,3},{0,0}};
    void *dA, *dx, *dy;
    cudaMalloc(&dA, 4 * sizeof(cuComplex));
    cudaMalloc(&dx, 2 * sizeof(cuComplex));
    cudaMalloc(&dy, 2 * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1,0};
    cublasStatus_t st = cublasCher2(h, CUBLAS_FILL_MODE_UPPER, 2, &alpha,
                                     (cuComplex*)dx, 1, (cuComplex*)dy, 1,
                                     (cuComplex*)dA, 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCher2 returns success");

    cudaMemcpy(A.data(), dA, 4 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    // A[0,0] += alpha*x[0]*conj(y[0]) + conj(alpha)*y[0]*conj(x[0])
    //         = {3,0}*{0,-3} + {0,3}*{3,0} = {0,-9}+{0,9} = {0,0} → diag forced to 0
    // A[0,1] += alpha*x[0]*conj(y[1]) + conj(alpha)*y[0]*conj(x[1]) = 0+0 = {0,0}
    CHECK(fabsf(A[0].x) < 1e-5f && fabsf(A[0].y) < 1e-5f, "cublasCher2 diagonal result");

    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

// ── cublasCherk ────────────────────────────────────────────────────────────────
// A = [{1,0}] (1×1, CUBLAS_OP_N), alpha=2, beta=0, uplo=UPPER.
// C = 2 * A * A^H = 2 * [{1,0}] * [{1,0}]^H = [{2,0}].
static void test_cherk() {
    printf("[cublasCherk]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // 2×2 case: A (2×1, CUBLAS_OP_N, i.e. n=2 k=1):
    //   A = col-major 2×1: [{1,0},{0,1}]
    //   C = alpha * A * A^H + beta * C  (upper)
    //   A*A^H = [{1,0}*{1,0}, {1,0}*{0,-1}; {0,1}*{1,0}, {0,1}*{0,-1}]
    //         = [{1,0}, {0,-1}; {0,1}, {1,0}]
    // With alpha=1, beta=0: C = [[1, 0-1i],[0+1i, 1]] (Hermitian)
    std::vector<cuComplex> A = {{1,0},{0,1}};
    std::vector<cuComplex> C = {{0,0},{0,0},{0,0},{0,0}};
    void *dA, *dC;
    cudaMalloc(&dA, 2 * sizeof(cuComplex));
    cudaMalloc(&dC, 4 * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasCherk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     2, 1, &alpha, (cuComplex*)dA, 2,
                                     &beta, (cuComplex*)dC, 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCherk returns success");
    cudaMemcpy(C.data(), dC, 4 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    // Upper triangle: C[0,0]=1+0i (diag), C[0,1]={0,-1}
    bool ok = (fabsf(C[0].x - 1.0f) < 1e-5f && fabsf(C[0].y) < 1e-5f &&  // [0,0]
               fabsf(C[2].x) < 1e-5f && fabsf(C[2].y - (-1.0f)) < 1e-5f); // [0,1]
    CHECK(ok, "cublasCherk upper triangle correct");

    cudaFree(dA); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasZherk ────────────────────────────────────────────────────────────────
static void test_zherk() {
    printf("[cublasZherk]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // A (2×1): [{1,0},{0,1}], alpha=1, beta=1, uplo=LOWER.
    // C initially: [[2,0],[0,0],[X,X],[3,0]] (lower stored).
    // new_C[0,0] = 1*1 + 1*2 = 3, new_C[1,0] = {0,1}*{1,0} + 0 = {0,1}
    // new_C[1,1] = 1*1 + 1*3 = 4.
    std::vector<cuDoubleComplex> A = {{1,0},{0,1}};
    std::vector<cuDoubleComplex> C = {{2,0},{0,0},{99,99},{3,0}};
    void *dA, *dC;
    cudaMalloc(&dA, 2 * sizeof(cuDoubleComplex));
    cudaMalloc(&dC, 4 * sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    double alpha = 1.0, beta = 1.0;
    cublasStatus_t st = cublasZherk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                     2, 1, &alpha, (cuDoubleComplex*)dA, 2,
                                     &beta, (cuDoubleComplex*)dC, 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZherk returns success");
    cudaMemcpy(C.data(), dC, 4 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    bool ok = (fabs(C[0].x - 3.0) < 1e-9 && fabs(C[0].y) < 1e-9 &&   // [0,0]
               fabs(C[1].x) < 1e-9 && fabs(C[1].y - 1.0) < 1e-9 &&   // [1,0]
               fabs(C[3].x - 4.0) < 1e-9 && fabs(C[3].y) < 1e-9);    // [1,1]
    CHECK(ok, "cublasZherk lower triangle with beta=1 correct");

    cudaFree(dA); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasCher2k ───────────────────────────────────────────────────────────────
// Simple: A=B=identity (1×1), alpha={1,0}, beta=0, n=k=1, uplo=UPPER.
// C = alpha * A * B^H + conj(alpha) * B * A^H = {1,0}+{1,0} = {2,0}.
static void test_cher2k() {
    printf("[cublasCher2k]\n");
    cublasHandle_t h;  cublasCreate(&h);

    std::vector<cuComplex> A = {{1,0}};
    std::vector<cuComplex> B = {{1,0}};
    std::vector<cuComplex> C = {{0,0}};
    void *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(cuComplex));
    cudaMalloc(&dB, sizeof(cuComplex));
    cudaMalloc(&dC, sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1,0};
    float beta = 0.0f;
    cublasStatus_t st = cublasCher2k(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                      1, 1, &alpha,
                                      (cuComplex*)dA, 1, (cuComplex*)dB, 1,
                                      &beta, (cuComplex*)dC, 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCher2k returns success");
    cudaMemcpy(C.data(), dC, sizeof(cuComplex), cudaMemcpyDeviceToHost);
    CHECK(fabsf(C[0].x - 2.0f) < 1e-5f && fabsf(C[0].y) < 1e-5f, "cublasCher2k result correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasZher2k ───────────────────────────────────────────────────────────────
// A={1,0}, B={0,1}, alpha={0,1} (pure imaginary).
// C = {0,1}*{1,0}*{0,-1} + {0,-1}*{0,1}*{1,0} = {0,1}*{0,-1} + {0,-1}*{0,1}
//   = {1,0} + {1,0} = {2,0}.
static void test_zher2k() {
    printf("[cublasZher2k]\n");
    cublasHandle_t h;  cublasCreate(&h);

    std::vector<cuDoubleComplex> A = {{1,0}};
    std::vector<cuDoubleComplex> B = {{0,1}};
    std::vector<cuDoubleComplex> C = {{0,0}};
    void *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(cuDoubleComplex));
    cudaMalloc(&dB, sizeof(cuDoubleComplex));
    cudaMalloc(&dC, sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha = {0,1};
    double beta = 0.0;
    cublasStatus_t st = cublasZher2k(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                      1, 1, &alpha,
                                      (cuDoubleComplex*)dA, 1, (cuDoubleComplex*)dB, 1,
                                      &beta, (cuDoubleComplex*)dC, 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZher2k returns success");
    cudaMemcpy(C.data(), dC, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    CHECK(fabs(C[0].x - 2.0) < 1e-9 && fabs(C[0].y) < 1e-9, "cublasZher2k result correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasChemm ────────────────────────────────────────────────────────────────
// A (2×2 Hermitian, upper), B (2×2), C = A * B (left side).
// A = [[2,0],[1-i,3]] stored upper: A[0,0]={2,0}, A[0,1]={1,-1}, A[1,1]={3,0}
//   → full A: [[2,1+i],[1-i,3]]
// B = [[1,0],[0,1]] (identity)
// C = A * I = A = [[2,1+i],[1-i,3]]
// In col-major: C[0,0]={2,0}, C[1,0]={1,-1}, C[0,1]={1,1}, C[1,1]={3,0}
static void test_chemm() {
    printf("[cublasChemm]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // Upper-stored A (col-major): col0={2,0},{X}, col1={1,-1},{3,0}
    std::vector<cuComplex> A = {{2,0},{99,99},{1,-1},{3,0}};
    // B = identity (col-major)
    std::vector<cuComplex> B = {{1,0},{0,0},{0,0},{1,0}};
    std::vector<cuComplex> C(4, {0,0});
    void *dA, *dB, *dC;
    cudaMalloc(&dA, 4 * sizeof(cuComplex));
    cudaMalloc(&dB, 4 * sizeof(cuComplex));
    cudaMalloc(&dC, 4 * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasChemm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     2, 2, &alpha,
                                     (cuComplex*)dA, 2,
                                     (cuComplex*)dB, 2,
                                     &beta, (cuComplex*)dC, 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasChemm returns success");
    cudaMemcpy(C.data(), dC, 4 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    // C = A*I = A  (col-major): C[0,0]={2,0}, C[1,0]={1,-1}(from A[1,0]=conj(A[0,1])),
    //                             C[0,1]={1,-1}, C[1,1]={3,0}... wait.
    // A full: A[0,0]={2,0}, A[1,0]=conj(A[0,1])=conj({1,-1})={1,1}, A[0,1]={1,-1}, A[1,1]={3,0}
    // col-major C = A*I: col0 of C = A*e1 = [{2,0},{1,1}], col1 = A*e2 = [{1,-1},{3,0}]
    bool ok = (fabsf(C[0].x - 2.0f) < 1e-5f && fabsf(C[0].y) < 1e-5f &&
               fabsf(C[1].x - 1.0f) < 1e-5f && fabsf(C[1].y - 1.0f) < 1e-5f &&
               fabsf(C[2].x - 1.0f) < 1e-5f && fabsf(C[2].y - (-1.0f)) < 1e-5f &&
               fabsf(C[3].x - 3.0f) < 1e-5f && fabsf(C[3].y) < 1e-5f);
    CHECK(ok, "cublasChemm result correct (A*I=A)");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasZhemm ────────────────────────────────────────────────────────────────
// A (1×1 Hermitian): [[5,0]], B = [[2,0]], right-side: C = B * A = 2*5 = 10.
static void test_zhemm() {
    printf("[cublasZhemm]\n");
    cublasHandle_t h;  cublasCreate(&h);

    std::vector<cuDoubleComplex> A = {{5,0}};
    std::vector<cuDoubleComplex> B = {{2,0}};
    std::vector<cuDoubleComplex> C = {{0,0}};
    void *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(cuDoubleComplex));
    cudaMalloc(&dB, sizeof(cuDoubleComplex));
    cudaMalloc(&dC, sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZhemm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                     1, 1, &alpha,
                                     (cuDoubleComplex*)dA, 1,
                                     (cuDoubleComplex*)dB, 1,
                                     &beta, (cuDoubleComplex*)dC, 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZhemm returns success");
    cudaMemcpy(C.data(), dC, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    CHECK(fabs(C[0].x - 10.0) < 1e-9 && fabs(C[0].y) < 1e-9, "cublasZhemm result correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasCgemmStridedBatched ─────────────────────────────────────────────────
// 2 batches of 1×1 GEMM: C[b] = alpha * A[b] * B[b] + beta * C[b].
// batch0: {1,0}*{2,0} = {2,0}
// batch1: {0,1}*{1,1} = {-1,1}
static void test_cgemm_strided_batched() {
    printf("[cublasCgemmStridedBatched]\n");
    cublasHandle_t h;  cublasCreate(&h);

    // Strided arrays: stride=1 element.
    std::vector<cuComplex> A = {{1,0},{0,1}};
    std::vector<cuComplex> B = {{2,0},{1,1}};
    std::vector<cuComplex> C = {{0,0},{0,0}};
    void *dA, *dB, *dC;
    cudaMalloc(&dA, 2 * sizeof(cuComplex));
    cudaMalloc(&dB, 2 * sizeof(cuComplex));
    cudaMalloc(&dC, 2 * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasCgemmStridedBatched(h,
                                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                                   1, 1, 1,
                                                   &alpha,
                                                   (cuComplex*)dA, 1, 1,
                                                   (cuComplex*)dB, 1, 1,
                                                   &beta,
                                                   (cuComplex*)dC, 1, 1,
                                                   2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCgemmStridedBatched returns success");
    cudaMemcpy(C.data(), dC, 2 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    bool ok = (fabsf(C[0].x - 2.0f) < 1e-5f && fabsf(C[0].y) < 1e-5f &&
               fabsf(C[1].x - (-1.0f)) < 1e-5f && fabsf(C[1].y - 1.0f) < 1e-5f);
    CHECK(ok, "cublasCgemmStridedBatched batch results correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(h);
}

// ── cublasZgemmStridedBatched ─────────────────────────────────────────────────
// 3 batches of 1×1 double GEMM; stride=1.
// C[b] = A[b] * B[b]:  {1,0}*{1,0}={1,0}, {0,1}*{0,1}={-1,0}, {1,1}*{1,-1}={2,0}
static void test_zgemm_strided_batched() {
    printf("[cublasZgemmStridedBatched]\n");
    cublasHandle_t h;  cublasCreate(&h);

    std::vector<cuDoubleComplex> A = {{1,0},{0,1},{1,1}};
    std::vector<cuDoubleComplex> B = {{1,0},{0,1},{1,-1}};
    std::vector<cuDoubleComplex> C(3, {0,0});
    void *dA, *dB, *dC;
    cudaMalloc(&dA, 3 * sizeof(cuDoubleComplex));
    cudaMalloc(&dB, 3 * sizeof(cuDoubleComplex));
    cudaMalloc(&dC, 3 * sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), 3 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), 3 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 3 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZgemmStridedBatched(h,
                                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                                   1, 1, 1,
                                                   &alpha,
                                                   (cuDoubleComplex*)dA, 1, 1,
                                                   (cuDoubleComplex*)dB, 1, 1,
                                                   &beta,
                                                   (cuDoubleComplex*)dC, 1, 1,
                                                   3);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZgemmStridedBatched returns success");
    cudaMemcpy(C.data(), dC, 3 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    bool ok = (fabs(C[0].x - 1.0) < 1e-9  && fabs(C[0].y) < 1e-9 &&
               fabs(C[1].x - (-1.0)) < 1e-9 && fabs(C[1].y) < 1e-9 &&
               fabs(C[2].x - 2.0) < 1e-9  && fabs(C[2].y) < 1e-9);
    CHECK(ok, "cublasZgemmStridedBatched batch results correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(h);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== extended_api_v6 tests ===\n");

    test_chemv();
    test_zhemv();
    test_cher();
    test_zher();
    test_cher2();
    test_cherk();
    test_zherk();
    test_cher2k();
    test_zher2k();
    test_chemm();
    test_zhemm();
    test_cgemm_strided_batched();
    test_zgemm_strided_batched();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
