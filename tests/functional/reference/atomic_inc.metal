#include <metal_stdlib>

using namespace metal;

kernel void atomic_inc(device atomic_uint* counter [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    (void)gid;
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}
