#include "cuda_fp16.h"

#include <cassert>
#include <cmath>
#include <cstdio>

// Unit tests for host-side cuda_fp16.h __half type.

int main() {
    // float → half → float roundtrip for simple values
    const float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 65504.0f, -65504.0f, 1.0f / 1024.0f};
    const int n = static_cast<int>(sizeof(vals) / sizeof(vals[0]));

    for (int i = 0; i < n; ++i) {
        const __half h = __float2half(vals[i]);
        const float back = __half2float(h);
        // Allow 0.1% relative error (half has ~3 significant decimal digits)
        const float tol = std::fabs(vals[i]) * 0.001f + 1e-6f;
        if (std::fabs(back - vals[i]) > tol) {
            std::fprintf(stderr,
                         "FAIL: __float2half/half2float roundtrip: input=%g, output=%g, diff=%g\n",
                         static_cast<double>(vals[i]),
                         static_cast<double>(back),
                         static_cast<double>(std::fabs(back - vals[i])));
            return 1;
        }
    }

    // Basic arithmetic via operator overloads
    const __half a = __float2half(3.0f);
    const __half b = __float2half(2.0f);

    const float sum = __half2float(a + b);
    if (std::fabs(sum - 5.0f) > 0.01f) {
        std::fprintf(stderr, "FAIL: __half add: got %g, expected 5.0\n", static_cast<double>(sum));
        return 1;
    }

    const float diff = __half2float(a - b);
    if (std::fabs(diff - 1.0f) > 0.01f) {
        std::fprintf(stderr, "FAIL: __half sub: got %g, expected 1.0\n", static_cast<double>(diff));
        return 1;
    }

    const float prod = __half2float(a * b);
    if (std::fabs(prod - 6.0f) > 0.01f) {
        std::fprintf(stderr,
                     "FAIL: __half mul: got %g, expected 6.0\n",
                     static_cast<double>(prod));
        return 1;
    }

    // Comparison operators
    if (!(a > b)) {
        std::fprintf(stderr, "FAIL: __half operator> failed\n");
        return 1;
    }
    if (!(b < a)) {
        std::fprintf(stderr, "FAIL: __half operator< failed\n");
        return 1;
    }
    if (!(a == a)) {
        std::fprintf(stderr, "FAIL: __half operator== failed\n");
        return 1;
    }
    if (!(a != b)) {
        std::fprintf(stderr, "FAIL: __half operator!= failed\n");
        return 1;
    }

    // __hadd / __hmul functions
    const float hadd_val = __half2float(__hadd(a, b));
    if (std::fabs(hadd_val - 5.0f) > 0.01f) {
        std::fprintf(stderr,
                     "FAIL: __hadd: got %g, expected 5.0\n",
                     static_cast<double>(hadd_val));
        return 1;
    }

    std::printf("PASS: cuda_fp16.h host-side __half type (roundtrip, arithmetic, comparison)\n");
    return 0;
}
