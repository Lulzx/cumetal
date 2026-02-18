#include "cuda.h"

#include <cstdio>

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

    size_t free0 = 0;
    size_t total0 = 0;
    if (cuMemGetInfo(&free0, &total0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemGetInfo initial query failed\n");
        return 1;
    }

    if (total0 == 0 || free0 > total0) {
        std::fprintf(stderr, "FAIL: invalid initial memory report\n");
        return 1;
    }

    constexpr size_t kAllocSize = 1u << 20;
    CUdeviceptr ptr = 0;
    if (cuMemAlloc(&ptr, kAllocSize) != CUDA_SUCCESS || ptr == 0) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    size_t free1 = 0;
    size_t total1 = 0;
    if (cuMemGetInfo(&free1, &total1) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemGetInfo after allocation failed\n");
        return 1;
    }

    if (total1 != total0 || free1 > free0) {
        std::fprintf(stderr, "FAIL: memory report should be monotonic after allocation\n");
        return 1;
    }

    if ((free0 - free1) < kAllocSize) {
        std::fprintf(stderr, "FAIL: reported free memory did not account for allocation size\n");
        return 1;
    }

    if (cuMemFree(ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    size_t free2 = 0;
    size_t total2 = 0;
    if (cuMemGetInfo(&free2, &total2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemGetInfo after free failed\n");
        return 1;
    }

    if (total2 != total0 || free2 != free0) {
        std::fprintf(stderr, "FAIL: memory report should return to baseline after free\n");
        return 1;
    }

    if (cuMemGetInfo(nullptr, &total0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null free output pointer should fail\n");
        return 1;
    }

    if (cuMemGetInfo(&free0, nullptr) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null total output pointer should fail\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver memory info API behaves correctly\n");
    return 0;
}
