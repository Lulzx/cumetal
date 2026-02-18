#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTotalElements = 8192;
constexpr std::size_t kWindowOffset = 113;
constexpr std::size_t kWindowElements = 4096;
constexpr std::size_t kThreadsPerBlock = 256;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
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

    std::vector<float> host_x(kTotalElements);
    std::vector<float> host_y(kTotalElements);
    std::vector<float> host_out(kTotalElements, -99.0f);

    for (std::size_t i = 0; i < kTotalElements; ++i) {
        host_x[i] = static_cast<float>((i * 13) % 31) * 0.125f;
        host_y[i] = static_cast<float>((i * 7) % 29) * 0.375f;
    }

    void* dev_x_base = nullptr;
    void* dev_y_base = nullptr;
    void* dev_out_base = nullptr;

    if (cudaMalloc(&dev_x_base, kTotalElements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_y_base, kTotalElements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_out_base, kTotalElements * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_x_base, host_x.data(), kTotalElements * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_y_base, host_y.data(), kTotalElements * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    float alpha = 2.25f;

    auto* dev_x_offset = static_cast<unsigned char*>(dev_x_base) + kWindowOffset * sizeof(float);
    auto* dev_y_offset = static_cast<unsigned char*>(dev_y_base) + kWindowOffset * sizeof(float);
    auto* dev_out_offset = static_cast<unsigned char*>(dev_out_base) + kWindowOffset * sizeof(float);

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BYTES, static_cast<std::uint32_t>(sizeof(float))},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "saxpy",
        .arg_count = 4,
        .arg_info = kArgInfo,
    };

    void* arg_x = dev_x_offset;
    void* arg_y = dev_y_offset;
    void* arg_out = dev_out_offset;
    void* launch_args[] = {&arg_x, &arg_y, &arg_out, &alpha};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kWindowElements + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock),
                        1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_out.data(), dev_out_base, kTotalElements * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kTotalElements; ++i) {
        const bool in_window = (i >= kWindowOffset && i < kWindowOffset + kWindowElements);
        if (!in_window) {
            continue;
        }

        const float expected = alpha * host_x[i] + host_y[i];
        if (!nearly_equal(host_out[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    if (cudaFree(dev_x_base) != cudaSuccess || cudaFree(dev_y_base) != cudaSuccess ||
        cudaFree(dev_out_base) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: runtime saxpy with scalar arg and pointer offsets succeeded\n");
    return 0;
}
