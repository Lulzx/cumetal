#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Tile size matches the kernel constant TILE = 16.
constexpr std::uint32_t kTile = 16;
// Use a matrix size that spans 2 tiles in each dimension.
constexpr std::uint32_t kMatrixN = 32;

bool nearly_equal(float a, float b) { return std::fabs(a - b) < 1e-3f; }

std::size_t idx(std::uint32_t row, std::uint32_t col, std::uint32_t n) {
    return static_cast<std::size_t>(row) * n + col;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    const std::size_t element_count = static_cast<std::size_t>(kMatrixN) * kMatrixN;
    std::vector<float> host_a(element_count);
    std::vector<float> host_b(element_count);
    std::vector<float> host_c(element_count, 0.0f);
    std::vector<float> expected(element_count, 0.0f);

    for (std::uint32_t r = 0; r < kMatrixN; ++r) {
        for (std::uint32_t c = 0; c < kMatrixN; ++c) {
            host_a[idx(r, c, kMatrixN)] = static_cast<float>((r * 7 + c * 3) % 19) * 0.5f;
            host_b[idx(r, c, kMatrixN)] = static_cast<float>((r * 5 + c * 11) % 23) * 0.25f;
        }
    }

    for (std::uint32_t r = 0; r < kMatrixN; ++r) {
        for (std::uint32_t c = 0; c < kMatrixN; ++c) {
            float acc = 0.0f;
            for (std::uint32_t k = 0; k < kMatrixN; ++k) {
                acc += host_a[idx(r, k, kMatrixN)] * host_b[idx(k, c, kMatrixN)];
            }
            expected[idx(r, c, kMatrixN)] = acc;
        }
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    void* dev_n = nullptr;

    const std::size_t bytes = element_count * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess ||
        cudaMalloc(&dev_n, sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    std::uint32_t n = kMatrixN;
    if (cudaMemcpy(dev_n, &n, sizeof(n), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy n failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "matrix_mul_tiled",
        .arg_count = 4,
        .arg_info = kArgInfo,
    };

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* arg_n = dev_n;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c, &arg_n};

    // Each block is a 16×16 tile.
    const dim3 block_dim(kTile, kTile, 1);
    const dim3 grid_dim((kMatrixN + kTile - 1) / kTile, (kMatrixN + kTile - 1) / kTile, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::uint32_t r = 0; r < kMatrixN; ++r) {
        for (std::uint32_t c = 0; c < kMatrixN; ++c) {
            if (!nearly_equal(host_c[idx(r, c, kMatrixN)], expected[idx(r, c, kMatrixN)])) {
                std::fprintf(stderr,
                             "FAIL: mismatch at (%u,%u) got=%f expected=%f\n",
                             r,
                             c,
                             static_cast<double>(host_c[idx(r, c, kMatrixN)]),
                             static_cast<double>(expected[idx(r, c, kMatrixN)]));
                return 1;
            }
        }
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess || cudaFree(dev_n) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: tiled matrix multiply (shared memory, 16×16 tiles, %u×%u)\n", kMatrixN, kMatrixN);
    return 0;
}
