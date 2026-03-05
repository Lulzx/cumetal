#pragma once
// CuMetal CUB shim: WarpScan — warp-level prefix scan.

namespace cub {

template <typename T, int LOGICAL_WARP_THREADS = 32, int LEGACY_PTX_ARCH = 0>
class WarpScan {
public:
    struct TempStorage {
        T data[LOGICAL_WARP_THREADS];
    };

    explicit WarpScan(TempStorage& temp) : temp_(temp), lane_id_(0) {}
    WarpScan(TempStorage& temp, int lane_id) : temp_(temp), lane_id_(lane_id) {}

    void InclusiveSum(T input, T& output) {
        temp_.data[lane_id_] = input;
        if (lane_id_ == 0) {
            for (int i = 1; i < LOGICAL_WARP_THREADS; i++)
                temp_.data[i] = temp_.data[i - 1] + temp_.data[i];
        }
        output = temp_.data[lane_id_];
    }

    void ExclusiveSum(T input, T& output) {
        temp_.data[lane_id_] = input;
        if (lane_id_ == 0) {
            T running = T{};
            for (int i = 0; i < LOGICAL_WARP_THREADS; i++) {
                T val = temp_.data[i];
                temp_.data[i] = running;
                running = running + val;
            }
        }
        output = temp_.data[lane_id_];
    }

    template <typename ScanOp>
    void InclusiveScan(T input, T& output, ScanOp op) {
        temp_.data[lane_id_] = input;
        if (lane_id_ == 0) {
            for (int i = 1; i < LOGICAL_WARP_THREADS; i++)
                temp_.data[i] = op(temp_.data[i - 1], temp_.data[i]);
        }
        output = temp_.data[lane_id_];
    }

    template <typename ScanOp>
    void ExclusiveScan(T input, T& output, T initial_value, ScanOp op) {
        temp_.data[lane_id_] = input;
        if (lane_id_ == 0) {
            T running = initial_value;
            for (int i = 0; i < LOGICAL_WARP_THREADS; i++) {
                T val = temp_.data[i];
                temp_.data[i] = running;
                running = op(running, val);
            }
        }
        output = temp_.data[lane_id_];
    }

private:
    TempStorage& temp_;
    int lane_id_;
};

} // namespace cub
