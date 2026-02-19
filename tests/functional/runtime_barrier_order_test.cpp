#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests that threadgroup_barrier (__syncthreads) provides the ordering guarantee:
// a write by thread 0 in shared memory is visible to all threads after the barrier
// (spec §5.3 / §10.3 "Synchronization: __syncthreads, barrier ordering guarantees").

namespace {

constexpr std::uint32_t kBlockSize = 64;
constexpr std::uint32_t kNumBlocks = 8;
constexpr std::uint32_t kN = kBlockSize * kNumBlocks;

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

    void* dev_out = nullptr;
    if (cudaMalloc(&dev_out, kN * sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "barrier_order_kernel",
        .arg_count = 1,
        .arg_info = kArgInfo,
    };

    void* launch_args[] = {&dev_out};

    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(kNumBlocks, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::vector<std::uint32_t> host_out(kN, 0);
    if (cudaMemcpy(host_out.data(), dev_out, kN * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    // Thread i in block k reads sentinel = k+1; output[k*kBlockSize+i] must equal k+1.
    for (std::uint32_t k = 0; k < kNumBlocks; ++k) {
        const std::uint32_t expected = k + 1u;
        for (std::uint32_t t = 0; t < kBlockSize; ++t) {
            const std::uint32_t got = host_out[k * kBlockSize + t];
            if (got != expected) {
                std::fprintf(stderr,
                             "FAIL: block %u thread %u: got %u expected %u "
                             "(barrier ordering violation)\n",
                             k, t, got, expected);
                return 1;
            }
        }
    }

    cudaFree(dev_out);

    std::printf(
        "PASS: __syncthreads barrier ordering guarantee (%u blocks x %u threads)\n",
        kNumBlocks,
        kBlockSize);
    return 0;
}
