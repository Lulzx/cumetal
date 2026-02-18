#include "cuda.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

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

    constexpr std::size_t kSize = 8192;
    CUdeviceptr dev = 0;
    if (cuMemAlloc(&dev, kSize) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    std::vector<std::uint8_t> host(kSize, 0u);

    if (cuMemsetD8(dev, 0xA5u, kSize) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemsetD8 failed\n");
        return 1;
    }

    if (cuMemcpyDtoH(host.data(), dev, kSize) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH after cuMemsetD8 failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kSize; ++i) {
        if (host[i] != 0xA5u) {
            std::fprintf(stderr, "FAIL: cuMemsetD8 mismatch at byte %zu\n", i);
            return 1;
        }
    }

    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamCreate failed\n");
        return 1;
    }

    if (cuMemsetD8Async(dev, 0x3Cu, kSize, stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemsetD8Async failed\n");
        return 1;
    }

    if (cuStreamSynchronize(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamSynchronize failed\n");
        return 1;
    }

    if (cuMemcpyDtoH(host.data(), dev, kSize) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH after cuMemsetD8Async failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kSize; ++i) {
        if (host[i] != 0x3Cu) {
            std::fprintf(stderr, "FAIL: cuMemsetD8Async mismatch at byte %zu\n", i);
            return 1;
        }
    }

    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy failed\n");
        return 1;
    }

    if (cuMemFree(dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuMemsetD8 and cuMemsetD8Async produced correct byte patterns\n");
    return 0;
}
