#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGetDeviceProperties(device=0) failed\n");
        return 1;
    }

    if (prop.name[0] == '\0') {
        std::fprintf(stderr, "FAIL: device name should not be empty\n");
        return 1;
    }

    if (prop.warpSize != 32) {
        std::fprintf(stderr, "FAIL: warpSize expected 32\n");
        return 1;
    }

    if (prop.totalGlobalMem == 0) {
        std::fprintf(stderr, "FAIL: totalGlobalMem should be non-zero\n");
        return 1;
    }

    if (prop.maxThreadsPerBlock <= 0 || prop.sharedMemPerBlock <= 0) {
        std::fprintf(stderr, "FAIL: block limits should be positive\n");
        return 1;
    }

    if (prop.major != 8 || prop.minor != 0) {
        std::fprintf(stderr, "FAIL: synthetic compute capability expected 8.0\n");
        return 1;
    }

    if (cudaGetDeviceProperties(nullptr, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null properties pointer should fail\n");
        return 1;
    }

    if (cudaGetDeviceProperties(&prop, 1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    std::printf("PASS: runtime device properties API behaves correctly\n");
    return 0;
}
