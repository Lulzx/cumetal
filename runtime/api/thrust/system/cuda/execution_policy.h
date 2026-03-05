#pragma once
// CuMetal thrust shim: thrust/system/cuda/execution_policy.h
// CUDA execution policy routes to the default (CPU) execution on UMA.
#include "../../execution_policy.h"

namespace thrust {
namespace system {
namespace cuda {
    using thrust::device;
} // namespace cuda
} // namespace system
} // namespace thrust
