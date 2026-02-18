#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 0) != cudaSuccess || value != 32) {
        std::fprintf(stderr, "FAIL: expected warp size 32\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: multiprocessor count should be positive\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: shared memory per block should be positive\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: unified addressing should be enabled\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrManagedMemory, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: managed memory should be enabled\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: concurrent managed access should be enabled\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(nullptr, cudaDevAttrWarpSize, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null output pointer should be rejected\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, 9999, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: unknown attribute should be rejected\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    std::printf("PASS: runtime device attribute API behaves correctly\n");
    return 0;
}
