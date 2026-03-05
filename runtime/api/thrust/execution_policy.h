#pragma once

// CuMetal thrust shim: execution policies.
// On Apple Silicon UMA, all policies route to CPU sequential execution.

namespace thrust {

struct device_execution_policy {};
struct host_execution_policy {};
struct sequential_execution_policy {};

// Standard execution policy tags
static constexpr device_execution_policy device;
static constexpr host_execution_policy host;
static constexpr sequential_execution_policy seq;

// cuda::par is the default device policy
namespace cuda_cub {
    static constexpr device_execution_policy par;
}

namespace system {
namespace cuda {
    using thrust::device_execution_policy;
}
}

} // namespace thrust
