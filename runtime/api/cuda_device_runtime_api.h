#pragma once
// CuMetal: cuda_device_runtime_api.h — stub for CUDA dynamic parallelism.
// Dynamic parallelism (device-side kernel launches) is not supported on Apple Silicon.
// This header allows code that conditionally includes it to compile;
// actual device-side cudaLaunchDevice calls will produce link errors.

#include "cuda_runtime.h"

// Provide the type so code that checks for it compiles
#ifdef __cplusplus
extern "C" {
#endif

// cudaLaunchDevice is the device-side kernel launch function.
// On CuMetal it is declared but not defined — linking against it will fail
// with a clear error message.
#ifndef __CUDA_ARCH__
static inline cudaError_t cudaLaunchDevice(void*, void**, dim3, dim3, unsigned int, cudaStream_t) {
    return cudaErrorUnknown;
}
#endif

#ifdef __cplusplus
}
#endif
