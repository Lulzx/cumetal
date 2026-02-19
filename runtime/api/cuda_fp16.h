#pragma once

// CuMetal cuda_fp16.h — half-precision float shim for Apple Silicon (UMA).
//
// On the host side, __half is a 16-bit IEEE 754 float stored as uint16_t.
// On the device side (when compiled by clang --cuda-gpu-arch via cumetalc),
// it is the native _Float16 type which Metal/AIR supports natively.
//
// Spec §8: "Half-precision atomics: Software emulation via CAS loop"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus

// ── Host-side __half (not compiled with CUDA device target) ─────────────────
#if !(defined(__clang__) && defined(__CUDA__))

struct __half {
    uint16_t __x;

    __half() = default;

    // Conversion from float
    explicit __half(float f) {
        // IEEE 754 float-to-half conversion
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
        uint32_t mant = bits & 0x7fffffu;
        if (exp <= 0) {
            __x = static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            __x = static_cast<uint16_t>(sign | 0x7c00u);
        } else {
            __x = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
        }
    }

    // Conversion to float
    explicit operator float() const {
        uint32_t sign = (__x >> 15) & 1u;
        uint32_t exp = (__x >> 10) & 0x1fu;
        uint32_t mant = __x & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            bits = (sign << 31);
        } else if (exp == 31) {
            bits = (sign << 31) | 0x7f800000u | (mant << 13);
        } else {
            bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
};

using __half2 = struct { __half x, y; };

inline __half __float2half(float f) { return __half(f); }
inline __half __float2half_rn(float f) { return __half(f); }
inline float __half2float(const __half& h) { return static_cast<float>(h); }

inline bool __hgt(const __half& a, const __half& b) {
    return static_cast<float>(a) > static_cast<float>(b);
}
inline bool __hlt(const __half& a, const __half& b) {
    return static_cast<float>(a) < static_cast<float>(b);
}
inline bool __heq(const __half& a, const __half& b) { return a.__x == b.__x; }
inline bool __hne(const __half& a, const __half& b) { return a.__x != b.__x; }

inline __half __hadd(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) + static_cast<float>(b));
}
inline __half __hmul(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) * static_cast<float>(b));
}
inline __half __hsub(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) - static_cast<float>(b));
}
inline __half __hdiv(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) / static_cast<float>(b));
}

inline __half operator+(const __half& a, const __half& b) { return __hadd(a, b); }
inline __half operator-(const __half& a, const __half& b) { return __hsub(a, b); }
inline __half operator*(const __half& a, const __half& b) { return __hmul(a, b); }
inline __half operator/(const __half& a, const __half& b) { return __hdiv(a, b); }
inline bool operator==(const __half& a, const __half& b) { return __heq(a, b); }
inline bool operator!=(const __half& a, const __half& b) { return __hne(a, b); }
inline bool operator>(const __half& a, const __half& b) { return __hgt(a, b); }
inline bool operator<(const __half& a, const __half& b) { return __hlt(a, b); }

#else  // Device code path (clang CUDA)

// When compiling device code, use the native _Float16 / __fp16 type.
// These are defined by the compiler; no additional typedef needed.
typedef _Float16 __half;

static __device__ __forceinline__ __half __float2half(float f) {
    return static_cast<__half>(f);
}
static __device__ __forceinline__ __half __float2half_rn(float f) {
    return static_cast<__half>(f);
}
static __device__ __forceinline__ float __half2float(__half h) {
    return static_cast<float>(h);
}

static __device__ __forceinline__ __half __hadd(__half a, __half b) { return a + b; }
static __device__ __forceinline__ __half __hmul(__half a, __half b) { return a * b; }
static __device__ __forceinline__ __half __hsub(__half a, __half b) { return a - b; }
static __device__ __forceinline__ __half __hdiv(__half a, __half b) { return a / b; }

// atomicAdd for __half via CAS loop (spec §8: "Software emulation via CAS loop").
// Uses the 32-bit word containing the 16-bit element for the CAS operation.
static __device__ __forceinline__ __half atomicAdd(__half* addr, __half val) {
    // Map the half address to the containing 32-bit word (must be 2-byte aligned).
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(addr) & ~static_cast<uintptr_t>(2));
    const bool high = (reinterpret_cast<uintptr_t>(addr) & 2) != 0;

    unsigned int assumed;
    unsigned int old = *base;
    do {
        assumed = old;
        unsigned short existing_bits = high ? (assumed >> 16) : (assumed & 0xffffu);
        __half existing;
        __builtin_memcpy(&existing, &existing_bits, 2);
        __half new_val = existing + val;
        unsigned short new_bits;
        __builtin_memcpy(&new_bits, &new_val, 2);
        unsigned int updated = high ? ((assumed & 0x0000ffffu) | (static_cast<unsigned int>(new_bits) << 16))
                                    : ((assumed & 0xffff0000u) | static_cast<unsigned int>(new_bits));
        old = __uAtomicCAS(base, assumed, updated);
    } while (old != assumed);

    unsigned short result_bits = high ? (old >> 16) : (old & 0xffffu);
    __half result;
    __builtin_memcpy(&result, &result_bits, 2);
    return result;
}

#endif  // device vs host

#endif  // __cplusplus
