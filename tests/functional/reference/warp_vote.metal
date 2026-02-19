#include <metal_stdlib>

using namespace metal;

// Warp vote/ballot test.
//
// Tests Metal simdgroup vote intrinsics:
//   simd_any    (CUDA __any_sync equivalent)
//   simd_all    (CUDA __all_sync equivalent)
//   simd_ballot (CUDA __ballot_sync equivalent)
//
// Partial-mask note: CuMetal maps all CUDA membermask variants conservatively
// to full-simdgroup operations.  Tests here use the full-participation result.
//
// Kernel layout (3 uint per thread):
//   [0]: simd_any  result (1 if true, 0 if false)
//   [1]: simd_all  result (1 if true, 0 if false)
//   [2]: ballot bits cast to uint (low 32 bits)
//
// Each simdgroup of 32 threads: even-indexed threads (gid % 2 == 0) have
// predicate = true.  All threads in the simdgroup see the same any/all/ballot.
kernel void warp_vote(device uint* output [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    const bool predicate = (gid % 2 == 0);

    const bool any_result = simd_any(predicate);
    const bool all_result = simd_all(predicate);

    // simd_ballot returns simd_vote; extract bits via explicit cast.
    const simd_vote ballot_val = simd_ballot(predicate);
    const uint ballot_uint =
        static_cast<uint>(static_cast<simd_vote::vote_t>(ballot_val));

    const uint base = gid * 3u;
    output[base + 0u] = any_result ? 1u : 0u;
    output[base + 1u] = all_result ? 1u : 0u;
    // For 32 threads where even lanes (0,2,4,...,30) have predicate=true:
    // ballot = 0x55555555
    output[base + 2u] = ballot_uint;
}
