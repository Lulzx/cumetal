#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    size_t free0 = 0;
    size_t total0 = 0;
    if (cudaMemGetInfo(&free0, &total0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: initial cudaMemGetInfo failed\n");
        return 1;
    }

    constexpr size_t kAllocSize = 1u << 20;
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, kAllocSize) != cudaSuccess || ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    size_t free1 = 0;
    size_t total1 = 0;
    if (cudaMemGetInfo(&free1, &total1) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemGetInfo after allocation failed\n");
        return 1;
    }
    if (total1 != total0 || free1 > free0) {
        std::fprintf(stderr, "FAIL: memory report should decrease after allocation\n");
        return 1;
    }

    if (cudaDeviceReset() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceReset failed\n");
        return 1;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != 0) {
        std::fprintf(stderr, "FAIL: device should reset to 0\n");
        return 1;
    }

    size_t free2 = 0;
    size_t total2 = 0;
    if (cudaMemGetInfo(&free2, &total2) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemGetInfo after reset failed\n");
        return 1;
    }
    if (total2 != total0 || free2 != free0) {
        std::fprintf(stderr, "FAIL: memory report should reset to baseline\n");
        return 1;
    }

    if (cudaFree(ptr) != cudaErrorInvalidDevicePointer) {
        std::fprintf(stderr, "FAIL: stale pointer should be invalid after reset\n");
        return 1;
    }

    if (cudaStreamSynchronize(stream) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: stale stream should be invalid after reset\n");
        return 1;
    }

    if (cudaDeviceReset() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: second cudaDeviceReset should succeed\n");
        return 1;
    }

    std::printf("PASS: runtime device reset API behaves correctly\n");
    return 0;
}
