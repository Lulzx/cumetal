#include <metal_stdlib>

using namespace metal;

// 2D grid indexing test.
// Each thread computes its linear index from (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
// + threadIdx.y * blockDim.x + threadIdx.x and writes it to output[linear_idx].
// Tests: 2D grid and block configurations, all six indexing intrinsics in 2D.
kernel void grid_2d(device uint* output [[buffer(0)]],
                    uint2 tid2  [[thread_position_in_threadgroup]],
                    uint2 gid2  [[threadgroup_position_in_grid]],
                    uint2 tpg2  [[threads_per_threadgroup]],
                    uint2 gpg2  [[threadgroups_per_grid]]) {
    const uint linear = (gid2.y * gpg2.x + gid2.x) * (tpg2.x * tpg2.y)
                        + tid2.y * tpg2.x + tid2.x;
    output[linear] = linear;
}
