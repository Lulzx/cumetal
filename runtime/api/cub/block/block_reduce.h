#pragma once
// CuMetal CUB shim: BlockReduce — sequential fallback for host-side compilation.
// In actual CUDA kernels translated to Metal, reductions use SIMD group ops.

#include <algorithm>
#include <cstring>

namespace cub {

enum BlockReduceAlgorithm {
    BLOCK_REDUCE_RAKING,
    BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
    BLOCK_REDUCE_WARP_REDUCTIONS
};

template <typename T, int BLOCK_DIM_X, BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockReduce {
public:
    struct TempStorage {
        T data[BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z];
    };

    explicit BlockReduce(TempStorage& temp) : temp_(temp), linear_tid_(0) {}
    BlockReduce(TempStorage& temp, int linear_tid) : temp_(temp), linear_tid_(linear_tid) {}

    // Full-tile reduce: all threads contribute one item
    template <typename ReduceOp>
    T Reduce(T input, ReduceOp op) {
        temp_.data[linear_tid_] = input;
        // Sequential fallback: thread 0 reduces all values
        if (linear_tid_ == 0) {
            T result = temp_.data[0];
            for (int i = 1; i < BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z; i++)
                result = op(result, temp_.data[i]);
            return result;
        }
        return input;
    }

    // Sum specialization
    T Sum(T input) {
        return Reduce(input, [](T a, T b) { return a + b; });
    }

    // Partial-tile reduce: only valid_items threads contribute
    template <typename ReduceOp>
    T Reduce(T input, ReduceOp op, int valid_items) {
        temp_.data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            T result = temp_.data[0];
            for (int i = 1; i < valid_items; i++)
                result = op(result, temp_.data[i]);
            return result;
        }
        return input;
    }

    T Sum(T input, int valid_items) {
        return Reduce(input, [](T a, T b) { return a + b; }, valid_items);
    }

private:
    TempStorage& temp_;
    int linear_tid_;
};

} // namespace cub
