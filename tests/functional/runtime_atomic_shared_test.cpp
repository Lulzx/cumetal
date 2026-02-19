#include "cuda_runtime.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

// Test: shared (threadgroup) memory atomics under contention.
//
// Each block of kBlockSize threads atomically increments a threadgroup counter.
// After the barrier, thread 0 adds the per-block total to the global output.
// With kGridSize blocks, the expected final value is kBlockSize * kGridSize.

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kGridSize  = 128;
constexpr uint32_t kExpected  = kBlockSize * kGridSize;

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

    uint32_t* d_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemset(d_output, 0, sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "atomic_shared",
        .arg_count     = 1,
        .arg_info      = kArgInfo,
    };

    void* arg_output = d_output;
    void* args[]     = {&arg_output};

    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(kGridSize, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    uint32_t h_output = 0;
    if (cudaMemcpy(&h_output, d_output, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H failed\n");
        return 1;
    }

    if (h_output != kExpected) {
        std::fprintf(stderr, "FAIL: output = %u, expected %u\n", h_output, kExpected);
        return 1;
    }

    if (cudaFree(d_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf(
        "PASS: shared memory atomics validated (%u blocks x %u threads = %u total)\n",
        kGridSize, kBlockSize, kExpected);
    return 0;
}
