#include <metal_stdlib>
using namespace metal;

kernel void matrix_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       device const uint* n_ptr [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint n = n_ptr[0];
    if (gid.x >= n || gid.y >= n) {
        return;
    }

    const uint row = gid.y;
    const uint col = gid.x;
    float acc = 0.0f;

    for (uint k = 0; k < n; ++k) {
        acc += a[row * n + k] * b[k * n + col];
    }

    c[row * n + col] = acc;
}
