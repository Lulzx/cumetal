#pragma once

#include <algorithm>

namespace thrust {

template <typename Iterator>
Iterator unique(Iterator first, Iterator last) {
    return std::unique(first, last);
}

template <typename Iterator, typename BinaryPred>
Iterator unique(Iterator first, Iterator last, BinaryPred pred) {
    return std::unique(first, last, pred);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator unique_copy(InputIterator first, InputIterator last,
                            OutputIterator result) {
    return std::unique_copy(first, last, result);
}

} // namespace thrust
