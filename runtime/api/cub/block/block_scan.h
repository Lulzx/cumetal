#pragma once
// CuMetal CUB shim: BlockScan — sequential fallback for host-side compilation.

#include <cstring>

namespace cub {

enum BlockScanAlgorithm {
    BLOCK_SCAN_RAKING,
    BLOCK_SCAN_RAKING_MEMOIZE,
    BLOCK_SCAN_WARP_SCANS
};

template <typename T, int BLOCK_DIM_X, BlockScanAlgorithm ALGORITHM = BLOCK_SCAN_RAKING,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockScan {
public:
    struct TempStorage {
        T data[BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z];
        T prefix;
    };

    explicit BlockScan(TempStorage& temp) : temp_(temp), linear_tid_(0) {}
    BlockScan(TempStorage& temp, int linear_tid) : temp_(temp), linear_tid_(linear_tid) {}

    // Exclusive prefix sum
    void ExclusiveSum(T input, T& output) {
        temp_.data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            constexpr int N = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
            T running = T{};
            for (int i = 0; i < N; i++) {
                T val = temp_.data[i];
                temp_.data[i] = running;
                running = running + val;
            }
        }
        output = temp_.data[linear_tid_];
    }

    // Exclusive scan with custom op and initial value
    template <typename ScanOp>
    void ExclusiveScan(T input, T& output, T initial_value, ScanOp op) {
        temp_.data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            constexpr int N = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
            T running = initial_value;
            for (int i = 0; i < N; i++) {
                T val = temp_.data[i];
                temp_.data[i] = running;
                running = op(running, val);
            }
        }
        output = temp_.data[linear_tid_];
    }

    // Inclusive prefix sum
    void InclusiveSum(T input, T& output) {
        temp_.data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            constexpr int N = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
            for (int i = 1; i < N; i++)
                temp_.data[i] = temp_.data[i - 1] + temp_.data[i];
        }
        output = temp_.data[linear_tid_];
    }

    // Inclusive scan with custom op
    template <typename ScanOp>
    void InclusiveScan(T input, T& output, ScanOp op) {
        temp_.data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            constexpr int N = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
            for (int i = 1; i < N; i++)
                temp_.data[i] = op(temp_.data[i - 1], temp_.data[i]);
        }
        output = temp_.data[linear_tid_];
    }

    // Exclusive sum with block aggregate
    void ExclusiveSum(T input, T& output, T& block_aggregate) {
        temp_.data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            constexpr int N = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
            T running = T{};
            for (int i = 0; i < N; i++) {
                T val = temp_.data[i];
                temp_.data[i] = running;
                running = running + val;
            }
            temp_.prefix = running;
        }
        output = temp_.data[linear_tid_];
        block_aggregate = temp_.prefix;
    }

private:
    TempStorage& temp_;
    int linear_tid_;
};

} // namespace cub
