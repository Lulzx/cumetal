#include "curand.h"

#include "cuda_runtime.h"

#include <mutex>
#include <new>
#include <random>

struct curandGenerator_st {
    std::mt19937_64 engine{0};
    std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
    std::mutex mutex;
};

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);

extern "C" {

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (rng_type != CURAND_RNG_PSEUDO_DEFAULT) {
        return CURAND_STATUS_TYPE_ERROR;
    }

    curandGenerator_t created = new (std::nothrow) curandGenerator_st();
    if (created == nullptr) {
        return CURAND_STATUS_ALLOCATION_FAILED;
    }

    *generator = created;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    delete generator;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                   unsigned long long seed) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    generator->engine.seed(seed);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* output_ptr, size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = generator->uniform(generator->engine);
    }
    return CURAND_STATUS_SUCCESS;
}

}  // extern "C"
