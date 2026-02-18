#include "cuda.h"

#include <cstdio>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0) != CUDA_SUCCESS || value != 32) {
        std::fprintf(stderr, "FAIL: expected warp size 32\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != CUDA_SUCCESS ||
        value <= 0) {
        std::fprintf(stderr, "FAIL: multiprocessor count should be positive\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0) != CUDA_SUCCESS ||
        value <= 0) {
        std::fprintf(stderr, "FAIL: shared memory per block should be positive\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0) != CUDA_SUCCESS ||
        value != 1) {
        std::fprintf(stderr, "FAIL: unified addressing should be enabled\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, 0) != CUDA_SUCCESS ||
        value != 1) {
        std::fprintf(stderr, "FAIL: managed memory should be enabled\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, 0) != CUDA_SUCCESS ||
        value != 1) {
        std::fprintf(stderr, "FAIL: concurrent managed access should be enabled\n");
        return 1;
    }

    if (cuDeviceGetAttribute(nullptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null output pointer should be rejected\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, static_cast<CUdevice_attribute>(9999), 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: unknown attribute should be rejected\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 1) != CUDA_ERROR_INVALID_DEVICE) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    std::printf("PASS: driver device attribute API behaves correctly\n");
    return 0;
}
