#include <metal_stdlib>
using namespace metal;

// Tests struct-by-value kernel argument passing (spec §6.5 step 5, §10.3 "Scalar arguments").
// The params struct is bound via setBytes (CUMETAL_ARG_BYTES) rather than a buffer pointer.
struct KernelParams {
    uint n;
    float scale;
    float offset;
};

kernel void struct_arg_kernel(device const float* input    [[buffer(0)]],
                              device float*       output   [[buffer(1)]],
                              constant KernelParams& params [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid >= params.n) {
        return;
    }
    output[tid] = input[tid] * params.scale + params.offset;
}
