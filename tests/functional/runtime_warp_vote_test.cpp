#include "cuda_runtime.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Test: warp vote and ballot intrinsics (spec §10.3 warp primitives).
//
// Launches 64 threads (2 simdgroups of 32) in a single block.
// Predicate: even-indexed threads → true, odd → false.
// Expected per-thread output (3 words per thread):
//   [0] any    = 1  (at least one even thread exists in each simdgroup)
//   [1] all    = 0  (not all threads have predicate; odd lanes have false)
//   [2] ballot = 0x55555555 (bits 0,2,4,...,30 set = even lanes)

namespace {

constexpr uint32_t kN            = 64;   // 2 simdgroups of 32
constexpr uint32_t kWordsPerThread = 3;
constexpr uint32_t kTotalWords   = kN * kWordsPerThread;
// Even lanes = bits 0,2,4,...,30 → 0x55555555
constexpr uint32_t kExpectedBallot = 0x55555555u;

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
    if (cudaMalloc(reinterpret_cast<void**>(&d_output),
                   kTotalWords * sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemset(d_output, 0xff, kTotalWords * sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "warp_vote",
        .arg_count     = 1,
        .arg_info      = kArgInfo,
    };

    void* arg_output = d_output;
    void* args[]     = {&arg_output};

    const dim3 block_dim(kN, 1, 1);
    const dim3 grid_dim(1, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::vector<uint32_t> h_output(kTotalWords, 0xffffffff);
    if (cudaMemcpy(h_output.data(), d_output, kTotalWords * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < kN; ++i) {
        const uint32_t any_val    = h_output[i * kWordsPerThread + 0];
        const uint32_t all_val    = h_output[i * kWordsPerThread + 1];
        const uint32_t ballot_val = h_output[i * kWordsPerThread + 2];

        if (any_val != 1u) {
            std::fprintf(stderr, "FAIL: thread %u: any=%u, expected 1\n", i, any_val);
            return 1;
        }
        if (all_val != 0u) {
            std::fprintf(stderr, "FAIL: thread %u: all=%u, expected 0\n", i, all_val);
            return 1;
        }
        if (ballot_val != kExpectedBallot) {
            std::fprintf(stderr, "FAIL: thread %u: ballot=0x%08x, expected 0x%08x\n",
                         i, ballot_val, kExpectedBallot);
            return 1;
        }
    }

    if (cudaFree(d_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: warp vote/ballot validated (%u threads, any=1 all=0 ballot=0x%08x)\n",
                kN, kExpectedBallot);
    return 0;
}
