#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct curandGenerator_st* curandGenerator_t;

typedef enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999,
} curandStatus_t;

typedef enum curandRngType {
    CURAND_RNG_PSEUDO_DEFAULT = 100,
} curandRngType_t;

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                   unsigned long long seed);
curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* output_ptr, size_t num);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                           double* output_ptr,
                                           size_t num);

#ifdef __cplusplus
}
#endif
