// CuMetal: cuda_runtime_api.h — forwarding header.
// Many CUDA programs include <cuda_runtime_api.h> for API-only declarations
// (no device code). Forward to our full cuda_runtime.h.
#pragma once
#include "cuda_runtime.h"
