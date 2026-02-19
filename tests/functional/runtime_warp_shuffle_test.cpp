#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Test: simd_shuffle broadcast from lane 0.
//
// Launch 64 threads in one block (two simdgroups of 32).
// input[i] = (float)i
// simd_shuffle(input[i], 0) returns the value held by lane 0 of the thread's
// simdgroup.  On Apple Silicon the simdgroup size is architecturally 32, so:
//   threads  0..31  (simdgroup 0, lane 0 = thread  0): output[i] = input[ 0] = 0.0
//   threads 32..63  (simdgroup 1, lane 0 = thread 32): output[i] = input[32] = 32.0

namespace {

constexpr uint32_t kN          = 64;
constexpr uint32_t kBlockSize  = 64;

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
        cudaMalloc(reinterpret_cast<void**>(&d_output), kN * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    // Prepare host input and copy to device.
    std::vector<float> h_input(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    if (cudaMemcpy(d_input, h_input.data(), kN * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H->D failed\n");
        return 1;
    }
    if (cudaMemset(d_output, 0, kN * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "warp_shuffle_broadcast",
        .arg_count     = 2,
        .arg_info      = kArgInfo,
    };

    void* arg_input  = d_input;
    void* arg_output = d_output;
    void* args[]     = {&arg_input, &arg_output};

    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(kN / kBlockSize, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::vector<float> h_output(kN, -1.0f);
    if (cudaMemcpy(h_output.data(), d_output, kN * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H failed\n");
        return 1;
    }

    // Verify: threads 0-31 should have output = 0.0, threads 32-63 should have output = 32.0.
    for (uint32_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>((i / 32u) * 32u);
        if (h_output[i] != expected) {
            std::fprintf(stderr, "FAIL: output[%u] = %.1f, expected %.1f\n", i, h_output[i],
                         expected);
            return 1;
        }
    }

    if (cudaFree(d_input) != cudaSuccess || cudaFree(d_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: warp shuffle broadcast correctness validated (%u threads)\n", kN);
    return 0;
}
