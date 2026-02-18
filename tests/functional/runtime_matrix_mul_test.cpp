#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kMatrixN = 32;
constexpr std::uint32_t kBlockX = 8;
constexpr std::uint32_t kBlockY = 8;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

std::size_t flat_index(std::uint32_t row, std::uint32_t col, std::uint32_t n) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(n) + static_cast<std::size_t>(col);
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

    for (std::uint32_t row = 0; row < kMatrixN; ++row) {
        for (std::uint32_t col = 0; col < kMatrixN; ++col) {
            const std::size_t idx = flat_index(row, col, kMatrixN);
            host_a[idx] = static_cast<float>((row * 7 + col * 3) % 19) * 0.5f;
            host_b[idx] = static_cast<float>((row * 5 + col * 11) % 23) * 0.25f;
        }
    }

    for (std::uint32_t row = 0; row < kMatrixN; ++row) {
        for (std::uint32_t col = 0; col < kMatrixN; ++col) {
            float acc = 0.0f;
            for (std::uint32_t k = 0; k < kMatrixN; ++k) {
                acc += host_a[flat_index(row, k, kMatrixN)] * host_b[flat_index(k, col, kMatrixN)];
            }
            expected[flat_index(row, col, kMatrixN)] = acc;
        }
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    void* dev_n = nullptr;

    const std::size_t bytes = element_count * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess || cudaMalloc(&dev_n, sizeof(std::uint32_t)) != cudaSuccess) {
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
        std::fprintf(stderr, "FAIL: cudaMemcpy n host->device failed\n");
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
        .kernel_name = "matrix_mul",
        .arg_count = 4,
        .arg_info = kArgInfo,
    };

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* arg_n = dev_n;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c, &arg_n};

    const dim3 block_dim(kBlockX, kBlockY, 1);
    const dim3 grid_dim((kMatrixN + kBlockX - 1) / kBlockX, (kMatrixN + kBlockY - 1) / kBlockY, 1);

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

    for (std::uint32_t row = 0; row < kMatrixN; ++row) {
        for (std::uint32_t col = 0; col < kMatrixN; ++col) {
            const std::size_t idx = flat_index(row, col, kMatrixN);
            if (!nearly_equal(host_c[idx], expected[idx])) {
                std::fprintf(stderr,
                             "FAIL: mismatch at (%u,%u) got=%f expected=%f\n",
                             row,
                             col,
                             static_cast<double>(host_c[idx]),
                             static_cast<double>(expected[idx]));
                return 1;
            }
        }
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess || cudaFree(dev_n) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: runtime matrix_mul produced correct %ux%u output\n", kMatrixN, kMatrixN);
    return 0;
}
