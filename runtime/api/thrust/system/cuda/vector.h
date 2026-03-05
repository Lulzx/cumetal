#pragma once
// CuMetal thrust shim: thrust/system/cuda/vector.h
// On Apple Silicon UMA, CUDA vectors are just device_vectors.
#include "../../device_vector.h"

namespace thrust {
namespace cuda_cub {
template <typename T>
using vector = thrust::device_vector<T>;
} // namespace cuda_cub

namespace system {
namespace cuda {
template <typename T>
using vector = thrust::device_vector<T>;
} // namespace cuda
} // namespace system
} // namespace thrust
