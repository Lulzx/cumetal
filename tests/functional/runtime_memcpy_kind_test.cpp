#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 4096;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    std::vector<float> host_src(kElementCount);
    std::vector<float> host_out(kElementCount, 0.0f);

    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_src[i] = static_cast<float>((i * 13) % 41) * 0.25f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;

    const std::size_t bytes = kElementCount * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_src.data(), bytes, cudaMemcpyDefault) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyDefault host->device failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_b, dev_a, bytes, cudaMemcpyDefault) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyDefault device->device failed\n");
        return 1;
    }
    if (cudaMemcpy(host_out.data(), dev_b, bytes, cudaMemcpyDefault) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyDefault device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        if (!nearly_equal(host_out[i], host_src[i])) {
            std::fprintf(stderr,
                         "FAIL: default memcpy mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(host_src[i]));
            return 1;
        }
    }

    if (cudaMemcpy(host_out.data(), host_src.data(), bytes, cudaMemcpyDeviceToHost) !=
        cudaErrorInvalidDevicePointer) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidDevicePointer for bad DtoH src\n");
        return 1;
    }
    if (cudaMemcpy(host_out.data(), dev_a, bytes, cudaMemcpyHostToDevice) !=
        cudaErrorInvalidDevicePointer) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidDevicePointer for bad HtoD dst\n");
        return 1;
    }
    if (cudaMemcpy(dev_b, host_src.data(), bytes, cudaMemcpyDeviceToDevice) !=
        cudaErrorInvalidDevicePointer) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidDevicePointer for bad DtoD src\n");
        return 1;
    }

    std::fill(host_out.begin(), host_out.end(), 0.0f);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    if (cudaMemcpyAsync(dev_b, host_src.data(), bytes, cudaMemcpyDefault, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyAsync default host->device failed\n");
        return 1;
    }
    if (cudaMemcpyAsync(host_out.data(), dev_b, bytes, cudaMemcpyDefault, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyAsync default device->host failed\n");
        return 1;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        if (!nearly_equal(host_out[i], host_src[i])) {
            std::fprintf(stderr,
                         "FAIL: async default memcpy mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(host_src[i]));
            return 1;
        }
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: cudaMemcpy kind validation and default inference are correct\n");
    return 0;
}
