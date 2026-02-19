#include <metal_stdlib>

using namespace metal;

// Broadcasts the value from lane 0 of each simdgroup to all lanes in that group.
// Equivalent to __shfl_sync(0xFFFFFFFF, val, 0) in CUDA.
kernel void warp_shuffle_broadcast(device const float* input  [[buffer(0)]],
                                   device float*       output [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]]) {
    float val = input[gid];
    float result = simd_shuffle(val, 0);
    output[gid] = result;
}
