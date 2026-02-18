#pragma once

#include "../cooperative_groups.h"

#if defined(__cplusplus)

namespace cooperative_groups {

template <int TileSize, typename T, typename BinaryOp>
__device__ __forceinline__ T reduce(const thread_block_tile<TileSize>& tile, T value, BinaryOp op) {
    __shared__ T shared[1024];
    const unsigned int linear_tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x +
                                    threadIdx.x;
    const unsigned int tile_base = tile.meta_group_rank() * TileSize;
    const unsigned int tile_rank = tile.thread_rank();

    shared[linear_tid] = value;
    __syncthreads();

    for (unsigned int offset = TileSize / 2u; offset > 0; offset >>= 1u) {
        if (tile_rank < offset) {
            shared[tile_base + tile_rank] =
                op(shared[tile_base + tile_rank], shared[tile_base + tile_rank + offset]);
        }
        __syncthreads();
    }

    return shared[tile_base];
}

}  // namespace cooperative_groups

#endif
