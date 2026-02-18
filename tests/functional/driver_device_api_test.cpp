#include "cuda.h"

#include <cstdio>

int main() {
    int driver_version = 0;
    if (cuDriverGetVersion(&driver_version) != CUDA_SUCCESS || driver_version <= 0) {
        std::fprintf(stderr, "FAIL: cuDriverGetVersion failed before init\n");
        return 1;
    }
    if (cuDriverGetVersion(nullptr) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuDriverGetVersion should reject null output pointer\n");
        return 1;
    }

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    int count = -1;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count != 1) {
        std::fprintf(stderr, "FAIL: cuDeviceGetCount expected 1\n");
        return 1;
    }

    CUdevice device = -1;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS || device != 0) {
        std::fprintf(stderr, "FAIL: cuDeviceGet(0) failed\n");
        return 1;
    }

    if (cuDeviceGet(&device, 1) != CUDA_ERROR_INVALID_DEVICE) {
        std::fprintf(stderr, "FAIL: cuDeviceGet(1) should fail\n");
        return 1;
    }

    std::printf("PASS: driver single-device APIs behave correctly\n");
    return 0;
}
