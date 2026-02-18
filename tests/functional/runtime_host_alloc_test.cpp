#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

constexpr std::size_t kElementCount = 1u << 14;
constexpr std::size_t kThreadsPerBlock = 256;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

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

    float* host_a = nullptr;
    float* host_b = nullptr;
    float* host_c = nullptr;

    const std::size_t bytes = kElementCount * sizeof(float);
    if (cudaHostAlloc(reinterpret_cast<void**>(&host_a), bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&host_b), bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&host_c), bytes, cudaHostAllocDefault) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaHostAlloc failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 13) % 37) * 0.25f;
        host_b[i] = static_cast<float>((i * 5) % 29) * 1.5f;
        host_c[i] = 0.0f;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "vector_add",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    void* arg_a = host_a;
    void* arg_b = host_b;
    void* arg_c = host_c;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock),
                        1,
                        1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    if (cudaFreeHost(host_a) != cudaSuccess || cudaFreeHost(host_b) != cudaSuccess ||
        cudaFreeHost(host_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFreeHost failed\n");
        return 1;
    }

    std::printf("PASS: cudaHostAlloc buffers are launchable as kernel arguments\n");
    return 0;
}
