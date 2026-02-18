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

    CUcontext ctx1 = nullptr;
    CUcontext ctx2 = nullptr;
    if (cuCtxCreate(&ctx1, 0, device) != CUDA_SUCCESS || ctx1 == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate(ctx1) failed\n");
        return 1;
    }
    if (cuCtxCreate(&ctx2, 0, device) != CUDA_SUCCESS || ctx2 == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate(ctx2) failed\n");
        return 1;
    }

    CUcontext current = reinterpret_cast<CUcontext>(0x1);
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx1) {
        std::fprintf(stderr, "FAIL: expected ctx1 as initial current context\n");
        return 1;
    }
    CUdevice current_device = -1;
    if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS || current_device != device) {
        std::fprintf(stderr, "FAIL: expected cuCtxGetDevice to report active device\n");
        return 1;
    }

    if (cuCtxSetCurrent(ctx2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(ctx2) failed\n");
        return 1;
    }
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx2) {
        std::fprintf(stderr, "FAIL: expected ctx2 as current context\n");
        return 1;
    }

    if (cuCtxSetCurrent(nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(nullptr) failed\n");
        return 1;
    }
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != nullptr) {
        std::fprintf(stderr, "FAIL: expected null current context\n");
        return 1;
    }
    if (cuCtxGetDevice(&current_device) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_CONTEXT from cuCtxGetDevice without context\n");
        return 1;
    }
    if (cuCtxSynchronize() != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_CONTEXT without current context\n");
        return 1;
    }

    if (cuCtxSetCurrent(ctx1) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(ctx1) failed\n");
        return 1;
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize with ctx1 failed\n");
        return 1;
    }
    if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS || current_device != device) {
        std::fprintf(stderr, "FAIL: cuCtxGetDevice should succeed after restoring current context\n");
        return 1;
    }

    if (cuCtxDestroy(ctx2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy(ctx2) failed\n");
        return 1;
    }
    if (cuCtxDestroy(ctx1) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy(ctx1) failed\n");
        return 1;
    }

    std::printf("PASS: driver context current-context APIs behave correctly\n");
    return 0;
}
