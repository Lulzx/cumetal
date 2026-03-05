#pragma once

// CuMetal thrust shim: inclusive/exclusive scan backed by CPU loops.

#include <functional>

namespace thrust {

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result) {
    if (first == last) return result;
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    T acc = *first;
    *result = acc;
    ++first; ++result;
    for (; first != last; ++first, ++result) {
        acc = acc + *first;
        *result = acc;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator, typename BinaryOp>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOp op) {
    if (first == last) return result;
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    T acc = *first;
    *result = acc;
    ++first; ++result;
    for (; first != last; ++first, ++result) {
        acc = op(acc, *first);
        *result = acc;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                               OutputIterator result) {
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    return exclusive_scan(first, last, result, T());
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                               OutputIterator result, T init) {
    for (; first != last; ++first, ++result) {
        *result = init;
        init = init + *first;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOp>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                               OutputIterator result, T init, BinaryOp op) {
    for (; first != last; ++first, ++result) {
        *result = init;
        init = op(init, *first);
    }
    return result;
}

} // namespace thrust
