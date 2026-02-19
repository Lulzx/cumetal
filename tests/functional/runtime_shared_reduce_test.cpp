#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Test: parallel reduction using threadgroup (shared) memory + barrier synchronization.
//
// Launch 256 threads in one block.  input[i] = 1.0f for all i.
// The shared_reduce kernel accumulates a single block using tree reduction:
//   stride = 128, 64, 32, 16, 8, 4, 2, 1  →  shared[0] = sum of 256 ones = 256.0f
// Expected: output[0] = 256.0f.

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kN        = kBlockSize;  // one block

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    float* d_input  = nullptr;
    float* d_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_input),  kN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_output), 1  * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    // All-ones input so the expected sum is exactly kN.
    std::vector<float> h_input(kN, 1.0f);
    if (cudaMemcpy(d_input, h_input.data(), kN * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H->D failed\n");
        return 1;
    }
    if (cudaMemset(d_output, 0, sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "shared_reduce",
        .arg_count     = 2,
        .arg_info      = kArgInfo,
    };

    void* arg_input  = d_input;
    void* arg_output = d_output;
    void* args[]     = {&arg_input, &arg_output};

    // One block of kBlockSize threads.  Grid dim is 1 group → output[0] = sum.
    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(1, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    float h_output = -1.0f;
    if (cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H failed\n");
        return 1;
    }

    const float expected = static_cast<float>(kN);
    if (h_output != expected) {
        std::fprintf(stderr, "FAIL: output = %.1f, expected %.1f\n", h_output, expected);
        return 1;
    }

    if (cudaFree(d_input) != cudaSuccess || cudaFree(d_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: shared memory reduction correctness validated (%u threads, sum=%.0f)\n",
                kN, h_output);
    return 0;
}
