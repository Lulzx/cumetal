#include "cuda.h"

#include <cstdio>
#include <cstdint>

int main() {
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

    constexpr size_t kBytes = 1024;
    CUdeviceptr ptr = 0;
    if (cuMemAllocManaged(&ptr, kBytes, 0) != CUDA_SUCCESS || ptr == 0) {
        std::fprintf(stderr, "FAIL: cuMemAllocManaged failed\n");
        return 1;
    }

    auto* bytes = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(ptr));
    bytes[0] = 0x2a;
    bytes[kBytes - 1] = 0x7f;
    if (bytes[0] != 0x2a || bytes[kBytes - 1] != 0x7f) {
        std::fprintf(stderr, "FAIL: managed allocation should be host-accessible\n");
        return 1;
    }

    if (cuMemFree(ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    if (cuMemAllocManaged(nullptr, kBytes, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null output pointer should fail\n");
        return 1;
    }

    if (cuMemAllocManaged(&ptr, 0, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: zero-size allocation should fail\n");
        return 1;
    }

    if (cuMemAllocManaged(&ptr, kBytes, 1) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: unsupported managed flags should fail\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver managed allocation API behaves correctly\n");
    return 0;
}
