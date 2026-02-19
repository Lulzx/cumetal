#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests that Apple Silicon warpSize == 32 and lane IDs are 0-31 (spec §7).
//
// Kernel: 2 blocks × 32 threads. Each thread writes its lane ID and simd_size().
// Expected:
//   - output[gid*2+0] == gid % 32      (lane ID within simdgroup)
//   - output[gid*2+1] == 32            (simd_size / warpSize)

static constexpr int kThreadsPerBlock = 32;
static constexpr int kBlocks = 2;
static constexpr int kN = kThreadsPerBlock * kBlocks;

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
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }

    const std::size_t out_words = kN * 2;
    const std::size_t out_bytes = out_words * sizeof(std::uint32_t);
    void* d_out = nullptr;
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }
    if (cudaMemset(d_out, 0, out_bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "warp_size_lane",
        .arg_count     = 1,
        .arg_info      = kArgInfo,
    };
    void* launch_args[] = {&d_out};
    const dim3 block_dim(kThreadsPerBlock, 1, 1);
    const dim3 grid_dim(kBlocks, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize\n");
        return 1;
    }

    std::vector<std::uint32_t> h_out(out_words, 0);
    if (cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy\n");
        return 1;
    }

    for (int i = 0; i < kN; ++i) {
        const std::uint32_t lane = h_out[i * 2 + 0];
        const std::uint32_t warp = h_out[i * 2 + 1];
        const std::uint32_t expected_lane = static_cast<std::uint32_t>(i % kThreadsPerBlock);
        if (lane != expected_lane) {
            std::fprintf(stderr, "FAIL: thread %d: laneid=%u expected %u\n", i, lane, expected_lane);
            cudaFree(d_out);
            return 1;
        }
        if (warp != 32u) {
            std::fprintf(stderr, "FAIL: thread %d: simd_size=%u expected 32\n", i, warp);
            cudaFree(d_out);
            return 1;
        }
    }

    cudaFree(d_out);
    std::printf("PASS: warpSize=32 and lane IDs verified for %d threads\n", kN);
    return 0;
}
