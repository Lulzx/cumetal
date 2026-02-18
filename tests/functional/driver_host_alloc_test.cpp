#include "cuda.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
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

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }

    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    CUmodule module = nullptr;
    if (cuModuleLoad(&module, metallib_path.c_str()) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleLoad failed\n");
        return 1;
    }

    CUfunction vector_add = nullptr;
    if (cuModuleGetFunction(&vector_add, module, "vector_add") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleGetFunction failed\n");
        return 1;
    }

    const std::size_t bytes = kElementCount * sizeof(float);
    float* host_a = nullptr;
    float* host_b = nullptr;
    float* host_c = nullptr;

    if (cuMemAllocHost(reinterpret_cast<void**>(&host_a), bytes) != CUDA_SUCCESS ||
        cuMemAllocHost(reinterpret_cast<void**>(&host_b), bytes) != CUDA_SUCCESS ||
        cuMemAllocHost(reinterpret_cast<void**>(&host_c), bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAllocHost failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 7) % 37) * 0.25f;
        host_b[i] = static_cast<float>((i * 5) % 29) * 1.5f;
        host_c[i] = 0.0f;
    }

    CUdeviceptr arg_a = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host_a));
    CUdeviceptr arg_b = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host_b));
    CUdeviceptr arg_c = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host_c));
    void* params[] = {&arg_a, &arg_b, &arg_c, nullptr};

    const unsigned int grid_x =
        static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock);

    if (cuLaunchKernel(vector_add,
                       grid_x,
                       1,
                       1,
                       static_cast<unsigned int>(kThreadsPerBlock),
                       1,
                       1,
                       0,
                       nullptr,
                       params,
                       nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuLaunchKernel failed\n");
        return 1;
    }

    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize failed\n");
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

    if (cuMemFreeHost(host_a) != CUDA_SUCCESS || cuMemFreeHost(host_b) != CUDA_SUCCESS ||
        cuMemFreeHost(host_c) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFreeHost failed\n");
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload failed\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuMemAllocHost buffers are launchable and writable\n");
    return 0;
}
