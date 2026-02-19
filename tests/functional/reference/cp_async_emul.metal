#include <metal_stdlib>
using namespace metal;

// Tests cp.async emulation: load values from global → threadgroup memory via an
// explicit synchronous copy + barrier (spec §5.1.1: cp.async lowered to plain
// ld+st + threadgroup_barrier; functional but not performance-equivalent).
//
// The kernel loads kN floats into a threadgroup tile, applies a scale factor,
// and writes the results back to global output.
kernel void cp_async_emul_kernel(device const float* input  [[buffer(0)]],
                                 device float*       output [[buffer(1)]],
                                 constant uint&      n      [[buffer(2)]],
                                 constant float&     scale  [[buffer(3)]],
                                 uint tid   [[thread_index_in_threadgroup]],
                                 uint gid   [[thread_position_in_grid]],
                                 uint gsize [[threads_per_threadgroup]]) {
    // Shared (threadgroup) tile — simulates the target of cp.async.
    threadgroup float tile[128];

    if (gid < n) {
        // Synchronous copy: global → threadgroup (emulates cp.async.ca.shared.global).
        tile[tid] = input[gid];
    } else {
        tile[tid] = 0.0f;
    }

    // Barrier: emulates cp.async.commit_group + cp.async.wait_all.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply scale using the threadgroup-resident value.
    if (gid < n) {
        output[gid] = tile[tid] * scale;
    }
}
