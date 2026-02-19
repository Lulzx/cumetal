#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests partial-mask warp shuffle semantics (spec §5.3 / §10.3).
//
// CuMetal conservatively lowers __shfl_sync(mask != 0xFFFFFFFF, …):
//   • Full-simdgroup shuffle is emitted.
//   • Lanes NOT in the mask read their own value (identity).
//
// Kernel: 2 blocks × 32 threads (1 simdgroup per block).
//   Lanes 0-15  (in mask 0x0000FFFF): get simd_broadcast from lane 0.
//   Lanes 16-31 (not in mask)        : keep their own input value.
//
// Input:  input[i] = (float)i
// Output: lane<16 → input[block_base], lane>=16 → input[i]

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

    std::vector<float> h_in(kN);
    for (int i = 0; i < kN; ++i) {
        h_in[i] = static_cast<float>(i);
    }
    std::vector<float> h_out(kN, 0.0f);

    void* d_in = nullptr;
    void* d_out = nullptr;
    const std::size_t bytes = kN * sizeof(float);
    if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_out, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }
    if (cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H→D\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "warp_partial_mask",
        .arg_count     = 2,
        .arg_info      = kArgInfo,
    };
    void* launch_args[] = {&d_in, &d_out};
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
    if (cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D→H\n");
        return 1;
    }

    // Verify: for each block b (0, 1), block_base = b * kThreadsPerBlock.
    //   lane = i % kThreadsPerBlock
    //   if lane < 16  → expected = h_in[block_base]  (lane 0 of block)
    //   if lane >= 16 → expected = h_in[i]           (own value / identity)
    for (int i = 0; i < kN; ++i) {
        const int block_base = (i / kThreadsPerBlock) * kThreadsPerBlock;
        const int lane = i % kThreadsPerBlock;
        const float expected = (lane < 16) ? h_in[block_base] : h_in[i];
        if (std::fabs(h_out[i] - expected) > 1e-6f) {
            std::fprintf(stderr,
                         "FAIL: output[%d] = %g, expected %g (lane=%d)\n",
                         i,
                         static_cast<double>(h_out[i]),
                         static_cast<double>(expected),
                         lane);
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    std::printf("PASS: warp partial-mask shuffle (%d threads, mask=0x0000FFFF)\n", kN);
    return 0;
}
