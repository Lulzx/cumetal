#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    cudaStream_t stream = nullptr;

    if (cudaStreamCreateWithFlags(&stream, cudaStreamDefault) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithFlags(default) failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy(default stream) failed\n");
        return 1;
    }

    stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithFlags(nonblocking) failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy(nonblocking stream) failed\n");
        return 1;
    }

    if (cudaStreamCreateWithFlags(&stream, 0x100u) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue for unsupported stream flag\n");
        return 1;
    }

    std::printf("PASS: cudaStreamCreateWithFlags validates supported flags\n");
    return 0;
}
