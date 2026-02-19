#include <metal_stdlib>

using namespace metal;

// 3D grid indexing test.
// Each thread computes its linear index across all three dimensions and writes
// it to output[linear_idx].  Tests: 3D grid and block configurations, all six
// indexing intrinsics in 3D.
//
// Linear index:
//   block_linear = gid.z * (gpg.y * gpg.x) + gid.y * gpg.x + gid.x
//   thread_local = tid.z * (tpg.y * tpg.x) + tid.y * tpg.x + tid.x
//   linear = block_linear * (tpg.x * tpg.y * tpg.z) + thread_local
kernel void grid_3d(device uint* output [[buffer(0)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 gid3  [[threadgroup_position_in_grid]],
                    uint3 tpg3  [[threads_per_threadgroup]],
                    uint3 gpg3  [[threadgroups_per_grid]]) {
    const uint block_stride = tpg3.x * tpg3.y * tpg3.z;
    const uint block_linear = (gid3.z * gpg3.y + gid3.y) * gpg3.x + gid3.x;
    const uint thread_local_idx = (tid3.z * tpg3.y + tid3.y) * tpg3.x + tid3.x;
    const uint linear = block_linear * block_stride + thread_local_idx;
    output[linear] = linear;
}
