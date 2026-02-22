#include "cuda_runtime.h"

#include <cstdio>

// Tests cudaDeviceGetStreamPriorityRange (spec §6.3, Metal has no priority queues).
// CuMetal returns 0,0 for both least and greatest priority.

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // Test null pointers are accepted (CUDA allows either to be null).
    int least = -1, greatest = -1;

    // Both non-null.
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetStreamPriorityRange failed\n");
        return 1;
    }
    // Metal has no stream priorities; both should be 0 (least == greatest == 0).
    if (least != 0 || greatest != 0) {
        std::fprintf(stderr, "FAIL: expected (0,0) priority range, got (%d,%d)\n", least, greatest);
        return 1;
    }

    // Null least pointer.
    greatest = -1;
    if (cudaDeviceGetStreamPriorityRange(nullptr, &greatest) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetStreamPriorityRange(null, ...) failed\n");
        return 1;
    }
    if (greatest != 0) {
        std::fprintf(stderr, "FAIL: greatest priority should be 0, got %d\n", greatest);
        return 1;
    }

    // Null greatest pointer.
    least = -1;
    if (cudaDeviceGetStreamPriorityRange(&least, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetStreamPriorityRange(..., null) failed\n");
        return 1;
    }
    if (least != 0) {
        std::fprintf(stderr, "FAIL: least priority should be 0, got %d\n", least);
        return 1;
    }

    // Verify cudaStreamCreateWithPriority ignores priority and creates a valid stream.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithPriority(&stream, cudaStreamDefault, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithPriority failed\n");
        return 1;
    }
    if (stream == nullptr) {
        std::fprintf(stderr, "FAIL: created stream is null\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: stream priority APIs correctly report Metal's flat priority model (0,0)\n");
    return 0;
}
