#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    int count = -1;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1) {
        std::fprintf(stderr, "FAIL: cudaGetDeviceCount expected 1\n");
        return 1;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != 0) {
        std::fprintf(stderr, "FAIL: cudaGetDevice expected 0\n");
        return 1;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaSetDevice(0) failed\n");
        return 1;
    }

    if (cudaSetDevice(1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaSetDevice(1) should fail\n");
        return 1;
    }

    std::printf("PASS: runtime single-device APIs behave correctly\n");
    return 0;
}
