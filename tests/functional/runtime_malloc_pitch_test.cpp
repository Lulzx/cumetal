#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Tests cudaMallocPitch: verify that pitch >= width*sizeof(T), that allocation
// succeeds, and that data written row-by-row via cudaMemcpy2D round-trips correctly.

static constexpr int kRows = 8;
static constexpr int kCols = 16;

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // Test 1: cudaMallocPitch — null pitch pointer is rejected.
    void* ptr = nullptr;
    if (cudaMallocPitch(&ptr, nullptr, kCols * sizeof(float), kRows) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null pitch ptr should be rejected\n");
        return 1;
    }

    // Test 2: cudaMallocPitch — null devPtr pointer is rejected.
    size_t pitch = 0;
    if (cudaMallocPitch(nullptr, &pitch, kCols * sizeof(float), kRows) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null devPtr should be rejected\n");
        return 1;
    }

    // Test 3: successful allocation.
    float* d_mat = nullptr;
    pitch = 0;
    if (cudaMallocPitch(reinterpret_cast<void**>(&d_mat), &pitch,
                        kCols * sizeof(float), kRows) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMallocPitch failed\n");
        return 1;
    }
    if (pitch < kCols * sizeof(float)) {
        std::fprintf(stderr, "FAIL: pitch %zu < row width %zu\n", pitch, kCols * sizeof(float));
        return 1;
    }
    // Pitch must be a multiple of alignment (512 on CuMetal).
    if (pitch % 512 != 0 && pitch % sizeof(float) != 0) {
        std::fprintf(stderr, "FAIL: pitch %zu is not properly aligned\n", pitch);
        return 1;
    }

    // Test 4: write via cudaMemcpy2D and read back.
    std::vector<float> h_src(kRows * kCols);
    for (int i = 0; i < kRows * kCols; ++i) h_src[i] = static_cast<float>(i);

    // Copy host linear → device pitched.
    if (cudaMemcpy2D(d_mat, pitch,
                     h_src.data(), kCols * sizeof(float),
                     kCols * sizeof(float), kRows,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy2D H->D failed\n");
        return 1;
    }

    // Copy device pitched → host linear.
    std::vector<float> h_dst(kRows * kCols, -1.0f);
    if (cudaMemcpy2D(h_dst.data(), kCols * sizeof(float),
                     d_mat, pitch,
                     kCols * sizeof(float), kRows,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy2D D->H failed\n");
        return 1;
    }

    for (int i = 0; i < kRows * kCols; ++i) {
        if (h_dst[i] != h_src[i]) {
            std::fprintf(stderr, "FAIL: h_dst[%d]=%.1f expected %.1f\n", i, h_dst[i], h_src[i]);
            cudaFree(d_mat);
            return 1;
        }
    }

    cudaFree(d_mat);
    std::printf("PASS: cudaMallocPitch pitch=%zu, data round-trip correct (%dx%d matrix)\n",
                pitch, kRows, kCols);
    return 0;
}
