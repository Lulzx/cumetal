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
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }
    if (curandSetStream(generator, stream) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandSetStream failed\n");
        return 1;
    }
    cudaStream_t queried_stream = nullptr;
    if (curandGetStream(generator, &queried_stream) != CURAND_STATUS_SUCCESS ||
        queried_stream != stream) {
        std::fprintf(stderr, "FAIL: curandGetStream mismatch\n");
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

    double* device_output_double = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_double), kCount * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for double output failed\n");
        return 1;
    }
    if (curandGenerateUniformDouble(generator, device_output_double, kCount) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateUniformDouble failed\n");
        return 1;
    }

    std::vector<double> host_double(kCount, 0.0);
    if (cudaMemcpy(host_double.data(),
                   device_output_double,
                   kCount * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for double output failed\n");
        return 1;
    }
    bool has_double_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        const double value = host_double[i];
        if (value < 0.0 || value > 1.0) {
            std::fprintf(stderr, "FAIL: generated double value out of range [0,1]: %f\n", value);
            return 1;
        }
        if (i > 0 && host_double[i] != host_double[i - 1]) {
            has_double_variation = true;
        }
    }
    if (!has_double_variation) {
        std::fprintf(stderr, "FAIL: generated double sequence has no variation\n");
        return 1;
    }

    float* device_output_normal = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_normal), kCount * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for normal output failed\n");
        return 1;
    }
    constexpr float kNormalMean = 1.5f;
    constexpr float kNormalStddev = 0.7f;
    if (curandGenerateNormal(generator, device_output_normal, kCount, kNormalMean, kNormalStddev) !=
        CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateNormal failed\n");
        return 1;
    }

    std::vector<float> host_normal(kCount, 0.0f);
    if (cudaMemcpy(host_normal.data(),
                   device_output_normal,
                   kCount * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for normal output failed\n");
        return 1;
    }
    double sum = 0.0;
    bool has_normal_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        sum += static_cast<double>(host_normal[i]);
        if (i > 0 && host_normal[i] != host_normal[i - 1]) {
            has_normal_variation = true;
        }
    }
    const double sample_mean = sum / static_cast<double>(kCount);
    if (!has_normal_variation) {
        std::fprintf(stderr, "FAIL: generated normal sequence has no variation\n");
        return 1;
    }
    if (sample_mean < 1.2 || sample_mean > 1.8) {
        std::fprintf(stderr,
                     "FAIL: normal sample mean out of expected range (got=%f expected~%f)\n",
                     sample_mean,
                     static_cast<double>(kNormalMean));
        return 1;
    }
    if (curandGenerateNormal(generator, device_output_normal, kCount, kNormalMean, 0.0f) !=
        CURAND_STATUS_OUT_OF_RANGE) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_OUT_OF_RANGE for zero stddev\n");
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
    if (curandSetStream(nullptr, stream) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null generator in setStream\n");
        return 1;
    }
    if (curandGetStream(generator, nullptr) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null out stream ptr\n");
        return 1;
    }

    if (cudaFree(device_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }
    if (cudaFree(device_output_double) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for double output failed\n");
        return 1;
    }
    if (cudaFree(device_output_normal) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for normal output failed\n");
        return 1;
    }
    if (curandDestroyGenerator(generator) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandDestroyGenerator failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuRAND generation + stream binding shim works on device allocations\n");
    return 0;
}
