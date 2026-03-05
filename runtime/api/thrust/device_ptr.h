#pragma once

// CuMetal thrust shim: device_ptr — thin wrapper around raw pointer.
// On Apple Silicon UMA, device pointers are host-accessible.

#include <cstddef>
#include <iterator>

namespace thrust {

template <typename T>
class device_ptr {
    T* ptr_ = nullptr;
public:
    typedef T value_type;
    typedef T& reference;
    typedef T* pointer;
    typedef ptrdiff_t difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    device_ptr() = default;
    explicit device_ptr(T* p) : ptr_(p) {}

    T* get() const { return ptr_; }
    operator T*() const { return ptr_; } // implicit conversion for UMA

    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T& operator[](ptrdiff_t i) const { return ptr_[i]; }

    device_ptr& operator++() { ++ptr_; return *this; }
    device_ptr operator++(int) { device_ptr t = *this; ++ptr_; return t; }
    device_ptr& operator--() { --ptr_; return *this; }
    device_ptr operator--(int) { device_ptr t = *this; --ptr_; return t; }
    device_ptr& operator+=(ptrdiff_t n) { ptr_ += n; return *this; }
    device_ptr& operator-=(ptrdiff_t n) { ptr_ -= n; return *this; }
    device_ptr operator+(ptrdiff_t n) const { return device_ptr(ptr_ + n); }
    device_ptr operator-(ptrdiff_t n) const { return device_ptr(ptr_ - n); }
    ptrdiff_t operator-(const device_ptr& o) const { return ptr_ - o.ptr_; }

    bool operator==(const device_ptr& o) const { return ptr_ == o.ptr_; }
    bool operator!=(const device_ptr& o) const { return ptr_ != o.ptr_; }
    bool operator<(const device_ptr& o) const { return ptr_ < o.ptr_; }
    bool operator>(const device_ptr& o) const { return ptr_ > o.ptr_; }
    bool operator<=(const device_ptr& o) const { return ptr_ <= o.ptr_; }
    bool operator>=(const device_ptr& o) const { return ptr_ >= o.ptr_; }
};

template <typename T>
device_ptr<T> device_pointer_cast(T* p) { return device_ptr<T>(p); }

template <typename T>
T* raw_pointer_cast(const device_ptr<T>& p) { return p.get(); }

} // namespace thrust
