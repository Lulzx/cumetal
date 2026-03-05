#pragma once

#include <utility>

namespace thrust {

using std::pair;
using std::make_pair;

template <typename Iterator, typename T>
pair<Iterator, Iterator> equal_range(Iterator first, Iterator last, const T& value) {
    auto lo = first, hi = last;
    // lower_bound
    auto lb = first;
    auto count = last - first;
    while (count > 0) {
        auto step = count / 2;
        auto mid = lb + step;
        if (*mid < value) { lb = ++mid; count -= step + 1; }
        else count = step;
    }
    // upper_bound
    auto ub = lb;
    count = last - lb;
    while (count > 0) {
        auto step = count / 2;
        auto mid = ub + step;
        if (!(value < *mid)) { ub = ++mid; count -= step + 1; }
        else count = step;
    }
    return {lb, ub};
}

} // namespace thrust
