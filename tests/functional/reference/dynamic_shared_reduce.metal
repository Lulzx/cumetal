#include <metal_stdlib>
using namespace metal;

// Tests dynamic (runtime-sized) threadgroup memory, set via setThreadgroupMemoryLength.
// The shared array size is not known at compile time — it is passed via the runtime's
// sharedMem parameter (spec §6.5 step 6, cudaLaunchKernel sharedMem argument).
// Each block reduces its input slice into output[group_id] using the dynamic shared buffer.
kernel void dynamic_shared_reduce(device const float* input  [[buffer(0)]],
                                  device float*       output [[buffer(1)]],
                                  threadgroup float*  shared [[threadgroup(0)]],
                                  uint tid      [[thread_index_in_threadgroup]],
                                  uint gid      [[thread_position_in_grid]],
                                  uint gsize    [[threads_per_threadgroup]],
                                  uint group_id [[threadgroup_position_in_grid]]) {
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = gsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = shared[0];
    }
}
