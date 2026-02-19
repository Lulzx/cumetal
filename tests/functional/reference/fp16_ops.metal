#include <metal_stdlib>

using namespace metal;

// Converts each float to half, adds 1.0h, converts back to float.
// Tests FP16 arithmetic lowering (CUDA __half operations → Metal half).
kernel void fp16_add(device const float* input  [[buffer(0)]],
                     device float*       output [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    half val = half(input[gid]);
    half result = val + half(1.0);
    output[gid] = float(result);
}
