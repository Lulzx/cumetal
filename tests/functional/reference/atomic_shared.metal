#include <metal_stdlib>

using namespace metal;

// Shared (threadgroup) memory atomic test.
//
// Each block has `kBlockSize` threads.  All threads atomically increment a
// threadgroup counter, then thread 0 of each block adds the per-block total
// to the global output counter.  Expected result: output == gridDim * blockDim.
kernel void atomic_shared(device atomic_uint* output [[buffer(0)]],
                          uint  tid [[thread_position_in_threadgroup]],
                          uint  tpg [[threads_per_threadgroup]]) {
    threadgroup atomic_uint tg_counter;
    if (tid == 0) {
        atomic_store_explicit(&tg_counter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    atomic_fetch_add_explicit(&tg_counter, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint block_count = atomic_load_explicit(&tg_counter, memory_order_relaxed);
        // Sanity-check: every thread in the block must have incremented.
        if (block_count == tpg) {
            atomic_fetch_add_explicit(output, block_count, memory_order_relaxed);
        }
    }
}
