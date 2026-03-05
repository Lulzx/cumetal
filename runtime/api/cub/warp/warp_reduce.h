#pragma once
// CuMetal CUB shim: WarpReduce — sequential fallback for host-side compilation.
// In Metal kernels, warp reductions map to SIMD group operations.

#include <algorithm>

namespace cub {

template <typename T, int LOGICAL_WARP_THREADS = 32, int LEGACY_PTX_ARCH = 0>
class WarpReduce {
public:
    struct TempStorage {
        T data[LOGICAL_WARP_THREADS];
    };

    explicit WarpReduce(TempStorage& temp) : temp_(temp), lane_id_(0) {}
    WarpReduce(TempStorage& temp, int lane_id) : temp_(temp), lane_id_(lane_id) {}

    template <typename ReduceOp>
    T Reduce(T input, ReduceOp op) {
        temp_.data[lane_id_] = input;
        if (lane_id_ == 0) {
            T result = temp_.data[0];
            for (int i = 1; i < LOGICAL_WARP_THREADS; i++)
                result = op(result, temp_.data[i]);
            return result;
        }
        return input;
    }

    T Sum(T input) {
        return Reduce(input, [](T a, T b) { return a + b; });
    }

    template <typename ReduceOp>
    T Reduce(T input, ReduceOp op, int valid_items) {
        temp_.data[lane_id_] = input;
        if (lane_id_ == 0) {
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
    int lane_id_;
};

} // namespace cub
