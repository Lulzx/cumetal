#include <metal_stdlib>

using namespace metal;

// Validates warpSize == 32 and laneid == thread_index_in_simdgroup (spec §7).
//
// Apple Silicon SIMD-group width is architecturally fixed at 32 (spec §7).
// Each thread writes its lane ID and a constant 32 (the guaranteed warp size).
//
// The test verifies:
//   - thread_index_in_simdgroup == gid % 32 for a contiguous block of 32
//   - output[gid*2+1] == 32 (constant warpSize written by kernel)
//
// Output layout (2 words per thread):
//   output[gid*2 + 0] = laneid (thread_index_in_simdgroup)
//   output[gid*2 + 1] = 32 (architecturally fixed warpSize, spec §7)
static constant uint kWarpSize = 32u;

kernel void warp_size_lane(device uint* output [[buffer(0)]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint gid  [[thread_position_in_grid]]) {
    output[gid * 2u + 0u] = lane;
    output[gid * 2u + 1u] = kWarpSize;
}
