#pragma once

#include <iterator>

namespace thrust {

template <typename T>
class counting_iterator {
    T value_;
public:
    typedef T value_type;
    typedef T reference;
    typedef T* pointer;
    typedef ptrdiff_t difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    counting_iterator() : value_(0) {}
    explicit counting_iterator(T v) : value_(v) {}

    T operator*() const { return value_; }
    T operator[](ptrdiff_t n) const { return value_ + (T)n; }

    counting_iterator& operator++() { ++value_; return *this; }
    counting_iterator operator++(int) { auto t = *this; ++value_; return t; }
    counting_iterator& operator--() { --value_; return *this; }
    counting_iterator& operator+=(ptrdiff_t n) { value_ += (T)n; return *this; }
    counting_iterator operator+(ptrdiff_t n) const { return counting_iterator(value_ + (T)n); }
    counting_iterator operator-(ptrdiff_t n) const { return counting_iterator(value_ - (T)n); }
    ptrdiff_t operator-(const counting_iterator& o) const { return (ptrdiff_t)(value_ - o.value_); }

    bool operator==(const counting_iterator& o) const { return value_ == o.value_; }
    bool operator!=(const counting_iterator& o) const { return value_ != o.value_; }
    bool operator<(const counting_iterator& o) const { return value_ < o.value_; }
};

template <typename T>
counting_iterator<T> make_counting_iterator(T v) { return counting_iterator<T>(v); }

} // namespace thrust
