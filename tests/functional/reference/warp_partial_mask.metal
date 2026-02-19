#include <metal_stdlib>

using namespace metal;

// Partial-mask warp shuffle test (spec §5.3 mask semantics).
//
// CuMetal lowers __shfl_sync with mask != 0xFFFFFFFF conservatively:
//   full-simdgroup shuffle is emitted, then lanes NOT in the mask
//   read their own value (identity) rather than the shuffled value.
//
// This kernel models that behavior:
//   - Lanes in mask (lane < 16): get simd_broadcast result from lane 0
//   - Lanes not in mask (lane >= 16): keep their own input value (identity)
//
// Expected output for thread at global index gid (32 threads per block):
//   if (lane = gid % 32) < 16  → output[gid] = input[block_base + 0]
//   if lane >= 16              → output[gid] = input[gid]
kernel void warp_partial_mask(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint gid  [[thread_position_in_grid]]) {
    float val = input[gid];
    // Simulate CuMetal conservative partial-mask lowering:
    // full-group broadcast, then select: mask ? broadcast_val : own_val
    float broadcast_val = simd_broadcast(val, 0);
    output[gid] = (lane < 16u) ? broadcast_val : val;
}
