#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    if (cudaProfilerStart() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaProfilerStart failed\n");
        return 1;
    }

    if (cudaProfilerStop() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaProfilerStop failed\n");
        return 1;
    }

    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: profiler APIs should not set runtime error state\n");
        return 1;
    }

    std::printf("PASS: runtime profiler APIs behave correctly\n");
    return 0;
}
