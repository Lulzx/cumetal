#include <metal_stdlib>

using namespace metal;

// Parallel reduction using threadgroup (shared) memory + barrier synchronization.
// Each threadgroup computes the sum of its input slice and writes it to output[group_id].
// Tests: threadgroup memory allocation, threadgroup_barrier ordering guarantees.
kernel void shared_reduce(device const float* input  [[buffer(0)]],
                          device float*       output [[buffer(1)]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint gid  [[thread_position_in_grid]],
                          uint gsize [[threads_per_threadgroup]],
                          uint group_id [[threadgroup_position_in_grid]]) {
    // Threadgroup (shared) memory for the reduction.
    threadgroup float shared[256];

    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: stride halves each step.
    for (uint stride = gsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = shared[0];
    }
}
