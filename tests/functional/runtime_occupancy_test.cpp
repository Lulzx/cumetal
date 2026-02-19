#include "cuda_runtime.h"

#include <cstdio>

// Tests spec §8 "Occupancy API" and related function/pointer attribute stubs.

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // --- cudaOccupancyMaxActiveBlocksPerMultiprocessor ---
    int numBlocks = -1;
    const void* dummy_func = reinterpret_cast<const void*>(0x1);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, dummy_func, 256, 0) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaOccupancyMaxActiveBlocksPerMultiprocessor failed\n");
        return 1;
    }
    if (numBlocks <= 0) {
        std::fprintf(stderr,
                     "FAIL: cudaOccupancyMaxActiveBlocksPerMultiprocessor returned %d (expected > 0)\n",
                     numBlocks);
        return 1;
    }

    // --- cudaOccupancyMaxPotentialBlockSize ---
    int minGridSize = -1;
    int blockSize = -1;
    if (cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dummy_func, 0, 0) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaOccupancyMaxPotentialBlockSize failed\n");
        return 1;
    }
    if (blockSize <= 0 || blockSize > 1024) {
        std::fprintf(stderr,
                     "FAIL: cudaOccupancyMaxPotentialBlockSize blockSize=%d (expected 1..1024)\n",
                     blockSize);
        return 1;
    }
    if (minGridSize <= 0) {
        std::fprintf(stderr,
                     "FAIL: cudaOccupancyMaxPotentialBlockSize minGridSize=%d (expected > 0)\n",
                     minGridSize);
        return 1;
    }
    // blockSizeLimit clamps: when limit=64, blockSize should be ≤64
    int limitedBlock = -1;
    int limitedGrid = -1;
    if (cudaOccupancyMaxPotentialBlockSize(&limitedGrid, &limitedBlock, dummy_func, 0, 64) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaOccupancyMaxPotentialBlockSize with limit failed\n");
        return 1;
    }
    if (limitedBlock > 64) {
        std::fprintf(stderr,
                     "FAIL: blockSizeLimit=64 but blockSize=%d\n",
                     limitedBlock);
        return 1;
    }

    // --- cudaFuncGetAttributes ---
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, dummy_func) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFuncGetAttributes failed\n");
        return 1;
    }
    if (attr.maxThreadsPerBlock <= 0) {
        std::fprintf(stderr,
                     "FAIL: maxThreadsPerBlock=%d (expected > 0)\n",
                     attr.maxThreadsPerBlock);
        return 1;
    }

    // --- cudaFuncSetCacheConfig (no-op, should succeed) ---
    if (cudaFuncSetCacheConfig(dummy_func, cudaFuncCachePreferL1) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFuncSetCacheConfig failed\n");
        return 1;
    }

    // --- cudaFuncSetSharedMemConfig (no-op, should succeed) ---
    if (cudaFuncSetSharedMemConfig(dummy_func, cudaSharedMemBankSizeEightByte) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFuncSetSharedMemConfig failed\n");
        return 1;
    }

    // --- cudaPointerGetAttributes ---
    // Host pointer: not in allocation table, should be cudaMemoryTypeHost
    cudaPointerAttributes pattr{};
    int local_val = 42;
    if (cudaPointerGetAttributes(&pattr, &local_val) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaPointerGetAttributes (host ptr) failed\n");
        return 1;
    }
    if (pattr.type != cudaMemoryTypeHost && pattr.type != cudaMemoryTypeUnregistered) {
        std::fprintf(stderr,
                     "FAIL: host ptr should be host/unregistered type, got %d\n",
                     static_cast<int>(pattr.type));
        return 1;
    }

    // Device pointer: allocated via cudaMalloc, should be managed/device
    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, 64) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    cudaPointerAttributes dattr{};
    if (cudaPointerGetAttributes(&dattr, dev_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaPointerGetAttributes (device ptr) failed\n");
        return 1;
    }
    if (dattr.type != cudaMemoryTypeManaged && dattr.type != cudaMemoryTypeDevice) {
        std::fprintf(stderr,
                     "FAIL: device ptr should be managed/device type, got %d\n",
                     static_cast<int>(dattr.type));
        return 1;
    }

    // --- cudaChooseDevice ---
    int chosen = -1;
    cudaDeviceProp prop{};
    prop.major = 8;
    prop.minor = 0;
    if (cudaChooseDevice(&chosen, &prop) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaChooseDevice failed\n");
        return 1;
    }
    if (chosen != 0) {
        std::fprintf(stderr, "FAIL: cudaChooseDevice returned %d (expected 0)\n", chosen);
        return 1;
    }

    cudaFree(dev_ptr);

    std::printf("PASS: occupancy API, func attrs, pointer attrs (spec §8)\n");
    return 0;
}
