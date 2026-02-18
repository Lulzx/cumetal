#include "cuda_runtime.h"
#include "curand.h"

#include <cstddef>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::size_t kCount = 4096;

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    curandGenerator_t generator = nullptr;
    if (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandCreateGenerator failed\n");
        return 1;
    }
    if (curandSetPseudoRandomGeneratorSeed(generator, 0xC0FFEEULL) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandSetPseudoRandomGeneratorSeed failed\n");
        return 1;
    }

    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), kCount * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (curandGenerateUniform(generator, device_output, kCount) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateUniform failed\n");
        return 1;
    }

    std::vector<float> host(kCount, 0.0f);
    if (cudaMemcpy(host.data(), device_output, kCount * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    float min_value = 1.0f;
    float max_value = 0.0f;
    bool has_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        const float value = host[i];
        if (value < 0.0f || value > 1.0f) {
            std::fprintf(stderr, "FAIL: generated value out of range [0,1]: %f\n", value);
            return 1;
        }
        if (i > 0 && host[i] != host[i - 1]) {
            has_variation = true;
        }
        if (value < min_value) {
            min_value = value;
        }
        if (value > max_value) {
            max_value = value;
        }
    }

    if (!has_variation) {
        std::fprintf(stderr, "FAIL: generated sequence has no variation\n");
        return 1;
    }
    if (min_value == max_value) {
        std::fprintf(stderr, "FAIL: generated sequence collapsed to a constant\n");
        return 1;
    }

    std::vector<float> host_output(kCount, 0.0f);
    if (curandGenerateUniform(generator, host_output.data(), kCount) != CURAND_STATUS_TYPE_ERROR) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_TYPE_ERROR for host output pointer\n");
        return 1;
    }

    if (curandCreateGenerator(nullptr, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null generator out ptr\n");
        return 1;
    }

    if (cudaFree(device_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }
    if (curandDestroyGenerator(generator) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandDestroyGenerator failed\n");
        return 1;
    }

    std::printf("PASS: cuRAND uniform generation shim works on device allocations\n");
    return 0;
}
