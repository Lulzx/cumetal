#include <metal_stdlib>
using namespace metal;

// Tests that threadgroup_barrier provides the __syncthreads ordering guarantee:
// all writes by any thread before the barrier are visible to all threads after it.
//
// Pattern (spec §5.3 / §10.3 "Synchronization: __syncthreads, barrier ordering"):
//   - Thread 0 writes a sentinel value into shared memory.
//   - All threads barrier.
//   - All threads read the sentinel from shared memory.
//   - They write it to global output.
//   - All output values must equal the sentinel.
kernel void barrier_order_kernel(device uint*       output   [[buffer(0)]],
                                 uint               tid      [[thread_index_in_threadgroup]],
                                 uint               gid      [[thread_position_in_grid]],
                                 uint               gsize    [[threads_per_threadgroup]],
                                 uint               group_id [[threadgroup_position_in_grid]]) {
    threadgroup uint sentinel = 0u;

    // Only thread 0 in each threadgroup writes the sentinel.
    if (tid == 0) {
        sentinel = group_id + 1u;  // unique per block, > 0
    }

    // Barrier: all threads must see thread 0's write.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread reads the sentinel — must equal group_id+1.
    output[gid] = sentinel;
}
