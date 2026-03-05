#pragma once
// CuMetal CUB shim: BlockExchange — data rearrangement between threads.

#include <cstring>

namespace cub {

template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD,
          bool WARP_TIME_SLICING = false, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockExchange {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

public:
    struct TempStorage {
        T data[TILE_ITEMS];
    };

    explicit BlockExchange(TempStorage& temp) : temp_(temp), linear_tid_(0) {}
    BlockExchange(TempStorage& temp, int linear_tid) : temp_(temp), linear_tid_(linear_tid) {}

    // Striped to blocked arrangement
    void StripedToBlocked(T (&items)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD]) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            temp_.data[linear_tid_ + i * BLOCK_THREADS] = items[i];
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            output[i] = temp_.data[linear_tid_ * ITEMS_PER_THREAD + i];
    }

    // Blocked to striped arrangement
    void BlockedToStriped(T (&items)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD]) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            temp_.data[linear_tid_ * ITEMS_PER_THREAD + i] = items[i];
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            output[i] = temp_.data[linear_tid_ + i * BLOCK_THREADS];
    }

    // In-place variants
    void StripedToBlocked(T (&items)[ITEMS_PER_THREAD]) {
        T tmp[ITEMS_PER_THREAD];
        StripedToBlocked(items, tmp);
        for (int i = 0; i < ITEMS_PER_THREAD; i++) items[i] = tmp[i];
    }

    void BlockedToStriped(T (&items)[ITEMS_PER_THREAD]) {
        T tmp[ITEMS_PER_THREAD];
        BlockedToStriped(items, tmp);
        for (int i = 0; i < ITEMS_PER_THREAD; i++) items[i] = tmp[i];
    }

private:
    TempStorage& temp_;
    int linear_tid_;
};

} // namespace cub
