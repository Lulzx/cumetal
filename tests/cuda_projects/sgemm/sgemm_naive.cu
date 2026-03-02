// sgemm_naive.cu — siboehm/SGEMM_CUDA kernel 1 (naive) run standalone
//
// Uses the unmodified sgemm_naive kernel from SGEMM_CUDA/src/kernels/1_naive.cuh
// with a simple host harness that verifies against a CPU matmul.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Naive SGEMM kernel (verbatim from SGEMM_CUDA/src/kernels/1_naive.cuh) ────

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                             const float *A, const float *B,
                             float beta, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────

static void sgemm_cpu(int M, int N, int K, float alpha,
                      const float *A, const float *B,
                      float beta, float *C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int k = 0; k < K; k++)
                acc += A[m*K+k] * B[k*N+n];
            C[m*N+n] = alpha * acc + beta * C[m*N+n];
        }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    const int M = 128, N = 128, K = 64;
    const float alpha = 1.0f, beta = 0.5f;
    const float TOL = 1e-3f;

    size_t sa = (size_t)M * K * sizeof(float);
    size_t sb = (size_t)K * N * sizeof(float);
    size_t sc = (size_t)M * N * sizeof(float);

    float *h_A  = (float*)malloc(sa);
    float *h_B  = (float*)malloc(sb);
    float *h_C  = (float*)malloc(sc);
    float *h_ref = (float*)malloc(sc);

    srand(42);
    for (int i = 0; i < M*K; i++) h_A[i]  = ((float)rand()/RAND_MAX)*2.f-1.f;
    for (int i = 0; i < K*N; i++) h_B[i]  = ((float)rand()/RAND_MAX)*2.f-1.f;
    for (int i = 0; i < M*N; i++) h_C[i]  = ((float)rand()/RAND_MAX)*0.1f;
    memcpy(h_ref, h_C, sc);

    // CPU reference
    sgemm_cpu(M, N, K, alpha, h_A, h_B, beta, h_ref);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sa); cudaMalloc(&d_B, sb); cudaMalloc(&d_C, sc);
    cudaMemcpy(d_A, h_A, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sc, cudaMemcpyHostToDevice);

    const int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sc, cudaMemcpyDeviceToHost);

    int errs = 0;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(h_C[i] - h_ref[i]);
        if (diff > TOL) {
            if (errs < 4)
                printf("  C[%d]: got %.6f ref %.6f diff %.2e\n", i, h_C[i], h_ref[i], diff);
            errs++;
        }
    }
    if (errs) {
        printf("FAIL: sgemm_naive: %d errors\n", errs);
        return 1;
    }
    printf("PASS: sgemm_naive (M=%d N=%d K=%d)\n", M, N, K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
