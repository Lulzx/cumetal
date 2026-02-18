#include "cuda.h"

#include <cstdio>
#include <cstring>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    char name[256] = {};
    if (cuDeviceGetName(name, static_cast<int>(sizeof(name)), 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGetName(device=0) failed\n");
        return 1;
    }

    if (name[0] == '\0') {
        std::fprintf(stderr, "FAIL: cuDeviceGetName returned an empty string\n");
        return 1;
    }

    size_t total_mem = 0;
    if (cuDeviceTotalMem(&total_mem, 0) != CUDA_SUCCESS || total_mem == 0) {
        std::fprintf(stderr, "FAIL: cuDeviceTotalMem(device=0) failed\n");
        return 1;
    }

    char tiny[4] = {'x', 'x', 'x', 'x'};
    if (cuDeviceGetName(tiny, static_cast<int>(sizeof(tiny)), 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGetName tiny buffer failed\n");
        return 1;
    }
    if (tiny[sizeof(tiny) - 1] != '\0') {
        std::fprintf(stderr, "FAIL: cuDeviceGetName should null-terminate output\n");
        return 1;
    }

    if (cuDeviceGetName(nullptr, static_cast<int>(sizeof(name)), 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null name pointer should fail\n");
        return 1;
    }

    if (cuDeviceGetName(name, 0, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: zero-length name buffer should fail\n");
        return 1;
    }

    if (cuDeviceGetName(name, static_cast<int>(sizeof(name)), 1) != CUDA_ERROR_INVALID_DEVICE) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    if (cuDeviceTotalMem(nullptr, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null byte-count pointer should fail\n");
        return 1;
    }

    if (cuDeviceTotalMem(&total_mem, 1) != CUDA_ERROR_INVALID_DEVICE) {
        std::fprintf(stderr, "FAIL: cuDeviceTotalMem should reject invalid device\n");
        return 1;
    }

    std::printf("PASS: driver device query APIs behave correctly\n");
    return 0;
}
