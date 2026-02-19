#include "cuda_runtime.h"

#include <cstdio>

// Tests cudaDeviceSetLimit / cudaDeviceGetLimit and cudaStreamCreateWithPriority (spec §6.3).

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // --- cudaDeviceGetLimit ---
    size_t stack_size = 0;
    if (cudaDeviceGetLimit(&stack_size, cudaLimitStackSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetLimit(cudaLimitStackSize) failed\n");
        return 1;
    }
    if (stack_size == 0) {
        std::fprintf(stderr, "FAIL: cudaLimitStackSize returned 0\n");
        return 1;
    }

    size_t printf_size = 0;
    if (cudaDeviceGetLimit(&printf_size, cudaLimitPrintfFifoSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetLimit(cudaLimitPrintfFifoSize) failed\n");
        return 1;
    }
    if (printf_size == 0) {
        std::fprintf(stderr, "FAIL: cudaLimitPrintfFifoSize returned 0\n");
        return 1;
    }

    size_t heap_size = 0;
    if (cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetLimit(cudaLimitMallocHeapSize) failed\n");
        return 1;
    }
    if (heap_size == 0) {
        std::fprintf(stderr, "FAIL: cudaLimitMallocHeapSize returned 0\n");
        return 1;
    }

    // --- cudaDeviceSetLimit (no-op; just verify it doesn't error) ---
    if (cudaDeviceSetLimit(cudaLimitStackSize, 2048) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSetLimit(cudaLimitStackSize) failed\n");
        return 1;
    }
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16 * 1024 * 1024) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSetLimit(cudaLimitMallocHeapSize) failed\n");
        return 1;
    }

    // --- cudaStreamCreateWithPriority ---
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithPriority(&stream, cudaStreamDefault, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithPriority failed\n");
        return 1;
    }
    if (stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithPriority returned null stream\n");
        return 1;
    }

    // Verify the stream works
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize on priority stream failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy on priority stream failed\n");
        return 1;
    }

    std::printf("PASS: device limits and stream priority APIs\n");
    return 0;
}
