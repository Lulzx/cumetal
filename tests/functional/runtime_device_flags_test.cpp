#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    unsigned int flags = 0xffffffffu;
    if (cudaGetDeviceFlags(&flags) != cudaSuccess || flags != cudaDeviceScheduleAuto) {
        std::fprintf(stderr, "FAIL: expected default device flags\n");
        return 1;
    }

    const unsigned int valid_flags = cudaDeviceScheduleSpin | cudaDeviceMapHost;
    if (cudaSetDeviceFlags(valid_flags) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaSetDeviceFlags(valid) failed\n");
        return 1;
    }
    if (cudaGetDeviceFlags(&flags) != cudaSuccess || flags != valid_flags) {
        std::fprintf(stderr, "FAIL: cudaGetDeviceFlags should return recently set flags\n");
        return 1;
    }

    if (cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceScheduleYield) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: conflicting schedule flags should fail\n");
        return 1;
    }

    if (cudaSetDeviceFlags(0x80u) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: unsupported device flags should fail\n");
        return 1;
    }

    if (cudaGetDeviceFlags(nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null output pointer should fail\n");
        return 1;
    }

    if (cudaDeviceReset() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceReset failed\n");
        return 1;
    }

    if (cudaGetDeviceFlags(&flags) != cudaSuccess || flags != cudaDeviceScheduleAuto) {
        std::fprintf(stderr, "FAIL: device flags should reset to default\n");
        return 1;
    }

    std::printf("PASS: runtime device flags APIs behave correctly\n");
    return 0;
}
