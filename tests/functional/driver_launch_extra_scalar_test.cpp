#include "cuda.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kMatrixN = 24;
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

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }

    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    CUmodule module = nullptr;
    if (cuModuleLoad(&module, metallib_path.c_str()) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleLoad failed\n");
        return 1;
    }

    CUfunction kernel = nullptr;
    if (cuModuleGetFunction(&kernel, module, "matrix_mul_scalar") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleGetFunction(matrix_mul_scalar) failed\n");
        return 1;
    }

    const std::size_t element_count = static_cast<std::size_t>(kMatrixN) * kMatrixN;
    const std::size_t bytes = element_count * sizeof(float);
    std::vector<float> host_a(element_count);
    std::vector<float> host_b(element_count);
    std::vector<float> host_c(element_count, 0.0f);
    std::vector<float> expected(element_count, 0.0f);

    for (std::uint32_t row = 0; row < kMatrixN; ++row) {
        for (std::uint32_t col = 0; col < kMatrixN; ++col) {
            const std::size_t idx = flat_index(row, col, kMatrixN);
            host_a[idx] = static_cast<float>((row * 7 + col * 5) % 19) * 0.75f;
            host_b[idx] = static_cast<float>((row * 11 + col * 3) % 17) * 0.5f;
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

    CUdeviceptr dev_a = 0;
    CUdeviceptr dev_b = 0;
    CUdeviceptr dev_c = 0;
    if (cuMemAlloc(&dev_a, bytes) != CUDA_SUCCESS || cuMemAlloc(&dev_b, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_c, bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    if (cuMemcpyHtoD(dev_a, host_a.data(), bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(dev_b, host_b.data(), bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyHtoD failed\n");
        return 1;
    }

    const std::uint64_t n_word = static_cast<std::uint64_t>(kMatrixN);
    std::uint64_t packed_args[] = {
        static_cast<std::uint64_t>(dev_a),
        static_cast<std::uint64_t>(dev_b),
        static_cast<std::uint64_t>(dev_c),
        n_word,
    };
    std::size_t packed_size = sizeof(packed_args);
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        packed_args,
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        &packed_size,
        CU_LAUNCH_PARAM_END,
    };

    if (cuLaunchKernel(kernel,
                       (kMatrixN + kBlockX - 1) / kBlockX,
                       (kMatrixN + kBlockY - 1) / kBlockY,
                       1,
                       kBlockX,
                       kBlockY,
                       1,
                       0,
                       nullptr,
                       nullptr,
                       extra) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuLaunchKernel(extra packed scalar arg) failed\n");
        return 1;
    }

    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize failed\n");
        return 1;
    }

    if (cuMemcpyDtoH(host_c.data(), dev_c, bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH failed\n");
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

    if (cuMemFree(dev_a) != CUDA_SUCCESS || cuMemFree(dev_b) != CUDA_SUCCESS ||
        cuMemFree(dev_c) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload failed\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuLaunchKernel extra path supports mixed pointer/scalar packed args\n");
    return 0;
}
