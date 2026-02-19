// Functional test: cuFFT C2C round-trip (forward + inverse = N * original).
// Uses a 1D C2C plan with a known input and verifies the inverse result equals
// N * input[i] within floating-point tolerance.

#include "cufft.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int N = 64;

    // Allocate host arrays.
    std::vector<cufftComplex> h_in(N), h_mid(N), h_out(N);
    for (int i = 0; i < N; ++i) {
        h_in[i].x = static_cast<float>(i % 7) + 1.0f;
        h_in[i].y = static_cast<float>(i % 5) * 0.5f;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }

    // Allocate device memory (UMA: device == host memory).
    cufftComplex* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_in), N * sizeof(cufftComplex)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_out), N * sizeof(cufftComplex)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }

    if (cudaMemcpy(d_in, h_in.data(), N * sizeof(cufftComplex), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H->D\n");
        return 1;
    }

    cufftHandle plan;
    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftPlan1d\n");
        return 1;
    }

    // Forward transform: d_in → d_out.
    if (cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftExecC2C FORWARD\n");
        return 1;
    }

    // Inverse transform: d_out → d_in (in-place inverse).
    if (cufftExecC2C(plan, d_out, d_in, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftExecC2C INVERSE\n");
        return 1;
    }

    if (cufftDestroy(plan) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftDestroy\n");
        return 1;
    }

    if (cudaMemcpy(h_out.data(), d_in, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H\n");
        return 1;
    }

    // After IFFT, result should be N * h_in[i] (cuFFT does not normalize).
    const float tol = static_cast<float>(N) * 1e-4f;
    for (int i = 0; i < N; ++i) {
        const float expected_re = static_cast<float>(N) * h_in[i].x;
        const float expected_im = static_cast<float>(N) * h_in[i].y;
        if (std::fabs(h_out[i].x - expected_re) > tol ||
            std::fabs(h_out[i].y - expected_im) > tol) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %d: got (%.4f,%.4f) expected (%.4f,%.4f)\n",
                         i, static_cast<double>(h_out[i].x), static_cast<double>(h_out[i].y),
                         static_cast<double>(expected_re), static_cast<double>(expected_im));
            return 1;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);

    std::printf("PASS: cuFFT C2C round-trip (N=%d)\n", N);
    return 0;
}
