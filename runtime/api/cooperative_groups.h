#pragma once

#include "cuda_runtime.h"

#if defined(__cplusplus)

namespace cooperative_groups {

struct thread_block {
    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return linear_tid();
    }

    __device__ __forceinline__ void sync() const {
        __syncthreads();
    }

private:
    __device__ __forceinline__ unsigned int linear_tid() const {
        return (threadIdx.z * (blockDim.y * blockDim.x)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    }
};

template <int TileSize>
struct thread_block_tile {
    static_assert(TileSize > 0, "TileSize must be positive");

    __device__ __forceinline__ unsigned int size() const { return TileSize; }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return linear_tid() % TileSize;
    }

    __device__ __forceinline__ unsigned int meta_group_rank() const {
        return linear_tid() / TileSize;
    }

    __device__ __forceinline__ unsigned int meta_group_size() const {
        return (block_size() + TileSize - 1u) / TileSize;
    }

    __device__ __forceinline__ void sync() const {
        __syncthreads();
    }

    // Warp-level shuffle within the tile.
    template <typename T>
    __device__ __forceinline__ T shfl(T val, unsigned int src_rank) const {
        return __shfl_sync(0xffffffffu, val, static_cast<int>(meta_group_rank() * TileSize + src_rank));
    }

    template <typename T>
    __device__ __forceinline__ T shfl_down(T val, unsigned int delta) const {
        return __shfl_down_sync(0xffffffffu, val, delta);
    }

    template <typename T>
    __device__ __forceinline__ T shfl_xor(T val, unsigned int lane_mask) const {
        return __shfl_xor_sync(0xffffffffu, val, static_cast<int>(lane_mask));
    }

    __device__ __forceinline__ int any(int predicate) const {
        return __nvvm_vote_any(predicate);
    }

    __device__ __forceinline__ int all(int predicate) const {
        return __nvvm_vote_all(predicate);
    }

    __device__ __forceinline__ unsigned int ballot(int predicate) const {
        return __nvvm_vote_ballot(predicate);
    }

private:
    __device__ __forceinline__ unsigned int linear_tid() const {
        return (threadIdx.z * (blockDim.y * blockDim.x)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    }

    __device__ __forceinline__ unsigned int block_size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
};

__device__ __forceinline__ thread_block this_thread_block() { return {}; }

template <int TileSize>
__device__ __forceinline__ thread_block_tile<TileSize> tiled_partition(const thread_block&) {
    return {};
}

template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct greater {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

template <typename T>
struct less {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};


}  // namespace cooperative_groups

#endif
