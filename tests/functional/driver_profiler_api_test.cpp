#include "cuda.h"

#include <cstdio>

int main() {
    if (cuProfilerStart() != CUDA_ERROR_NOT_INITIALIZED ||
        cuProfilerStop() != CUDA_ERROR_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: profiler APIs should require cuInit first\n");
        return 1;
    }

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    if (cuProfilerStart() != CUDA_ERROR_INVALID_CONTEXT ||
        cuProfilerStop() != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: profiler APIs should require a current context\n");
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

    if (cuProfilerStart() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuProfilerStart with active context failed\n");
        return 1;
    }

    if (cuProfilerStop() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuProfilerStop with active context failed\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver profiler APIs behave correctly\n");
    return 0;
}
