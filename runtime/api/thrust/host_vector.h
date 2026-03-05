#pragma once

// CuMetal thrust shim: host_vector is just std::vector.

#include "device_vector.h"
#include <vector>

namespace thrust {

template <typename T>
class host_vector : public std::vector<T> {
    using std::vector<T>::vector;
public:
    host_vector() = default;
    host_vector(const device_vector<T>& dv) : std::vector<T>(dv.size()) {
        cudaMemcpy(this->data(), dv.data(), dv.size() * sizeof(T), cudaMemcpyDeviceToHost);
    }
    host_vector& operator=(const device_vector<T>& dv) {
        this->resize(dv.size());
        cudaMemcpy(this->data(), dv.data(), dv.size() * sizeof(T), cudaMemcpyDeviceToHost);
        return *this;
    }
};

} // namespace thrust
