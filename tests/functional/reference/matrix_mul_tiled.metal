#include <metal_stdlib>
using namespace metal;

// Tiled matrix multiply using threadgroup (shared) memory.
// Loads 16×16 tiles from A and B into shared memory to reduce global-memory traffic.
// Tests: shared-memory allocation, two-barrier synchronization per tile step.
constant uint TILE = 16;

kernel void matrix_mul_tiled(device const float* a      [[buffer(0)]],
                             device const float* b      [[buffer(1)]],
                             device float*       c      [[buffer(2)]],
                             device const uint*  n_ptr  [[buffer(3)]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 bid [[threadgroup_position_in_grid]]) {
    const uint n = n_ptr[0];
    const uint row = bid.y * TILE + tid.y;
    const uint col = bid.x * TILE + tid.x;

    threadgroup float tile_a[16 * 16];
    threadgroup float tile_b[16 * 16];

    float acc = 0.0f;
    const uint num_tiles = (n + TILE - 1) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        const uint a_col = t * TILE + tid.x;
        const uint b_row = t * TILE + tid.y;

        tile_a[tid.y * TILE + tid.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[tid.y * TILE + tid.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k) {
            acc += tile_a[tid.y * TILE + k] * tile_b[k * TILE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}
