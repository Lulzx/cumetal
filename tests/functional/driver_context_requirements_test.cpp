#include "cuda.h"

#include <cstdio>

int main() {
    CUstream stream = nullptr;
    CUevent event = nullptr;
    CUdeviceptr device_ptr = 0;

    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_ERROR_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: cuStreamCreate should require cuInit\n");
        return 1;
    }
    if (cuEventCreate(&event, 0) != CUDA_ERROR_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: cuEventCreate should require cuInit\n");
        return 1;
    }
    if (cuMemAlloc(&device_ptr, 64) != CUDA_ERROR_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: cuMemAlloc should require cuInit\n");
        return 1;
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

    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: cuStreamCreate should require current context\n");
        return 1;
    }
    if (cuEventCreate(&event, 0) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: cuEventCreate should require current context\n");
        return 1;
    }
    if (cuMemAlloc(&device_ptr, 64) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: cuMemAlloc should require current context\n");
        return 1;
    }

    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS || context == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamCreate with context failed\n");
        return 1;
    }
    if (cuEventCreate(&event, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuEventCreate with context failed\n");
        return 1;
    }
    if (cuMemAlloc(&device_ptr, 64) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc with context failed\n");
        return 1;
    }

    if (cuMemFree(device_ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }
    if (cuEventDestroy(event) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuEventDestroy failed\n");
        return 1;
    }
    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy failed\n");
        return 1;
    }

    if (cuCtxSetCurrent(nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(nullptr) failed\n");
        return 1;
    }

    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: cuStreamCreate should fail with no current context\n");
        return 1;
    }
    if (cuMemAlloc(&device_ptr, 64) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: cuMemAlloc should fail with no current context\n");
        return 1;
    }

    if (cuCtxSetCurrent(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(context) failed\n");
        return 1;
    }
    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver APIs enforce init and current-context requirements\n");
    return 0;
}
