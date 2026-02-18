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

    CUstream stream = nullptr;

    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cuStreamCreate(default) failed\n");
        return 1;
    }
    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy(default stream) failed\n");
        return 1;
    }

    stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cuStreamCreate(nonblocking) failed\n");
        return 1;
    }
    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy(nonblocking stream) failed\n");
        return 1;
    }

    if (cuStreamCreate(&stream, 0x100u) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_VALUE for unsupported stream flag\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuStreamCreate validates supported flags\n");
    return 0;
}
