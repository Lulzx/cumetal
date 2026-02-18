#include "cuda_runtime.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

constexpr std::uint32_t kThreadsPerBlock = 256;
constexpr std::uint32_t kLaunchThreads = 1u << 18;
constexpr std::uint32_t kLaunchRepeats = 8;

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

    std::uint32_t* counter = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&counter), sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemset(counter, 0, sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "atomic_inc",
        .arg_count = 1,
        .arg_info = kArgInfo,
    };

    void* arg_counter = counter;
    void* args[] = {&arg_counter};

    const dim3 block_dim(kThreadsPerBlock, 1, 1);
    const dim3 grid_dim((kLaunchThreads + kThreadsPerBlock - 1) / kThreadsPerBlock, 1, 1);

    for (std::uint32_t i = 0; i < kLaunchRepeats; ++i) {
        if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaLaunchKernel failed on repeat %u\n", i);
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::uint32_t host_counter = 0;
    if (cudaMemcpy(&host_counter, counter, sizeof(host_counter), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    const std::uint32_t expected = kLaunchThreads * kLaunchRepeats;
    if (host_counter != expected) {
        std::fprintf(stderr, "FAIL: atomic mismatch (got=%u expected=%u)\n", host_counter, expected);
        return 1;
    }

    if (cudaFree(counter) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: atomic correctness validated (%u increments)\n", expected);
    return 0;
}
