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
        std::fprintf(stderr, "FAIL: cudaMemGetInfo initial query failed\n");
        return 1;
    }

    if (total0 == 0 || free0 > total0) {
        std::fprintf(stderr, "FAIL: invalid initial memory report\n");
        return 1;
    }

    constexpr size_t kAllocSize = 1u << 20;
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, kAllocSize) != cudaSuccess || ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    size_t free1 = 0;
    size_t total1 = 0;
    if (cudaMemGetInfo(&free1, &total1) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemGetInfo after allocation failed\n");
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

    if (cudaFree(ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    size_t free2 = 0;
    size_t total2 = 0;
    if (cudaMemGetInfo(&free2, &total2) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemGetInfo after free failed\n");
        return 1;
    }

    if (total2 != total0 || free2 != free0) {
        std::fprintf(stderr, "FAIL: memory report should return to baseline after free\n");
        return 1;
    }

    if (cudaMemGetInfo(nullptr, &total0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null free output pointer should fail\n");
        return 1;
    }

    if (cudaMemGetInfo(&free0, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null total output pointer should fail\n");
        return 1;
    }

    std::printf("PASS: runtime memory info API behaves correctly\n");
    return 0;
}
