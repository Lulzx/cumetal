#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests that a POD struct argument is correctly passed to a kernel by value
// (spec §6.5 step 5 "scalar/struct arguments ≤ 4 KB: setBytes").
// The Metal shader takes a `constant KernelParams& params [[buffer(2)]]` that
// holds n, scale, and offset.

namespace {

struct alignas(4) KernelParams {
    std::uint32_t n;
    float scale;
    float offset;
};

constexpr std::uint32_t kN = 64;

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

    std::vector<float> host_in(kN);
    std::vector<float> host_out(kN, 0.0f);
    for (std::uint32_t i = 0; i < kN; ++i) {
        host_in[i] = static_cast<float>(i);
    }

    void* dev_in = nullptr;
    void* dev_out = nullptr;
    const std::size_t bytes = kN * sizeof(float);

    if (cudaMalloc(&dev_in, bytes) != cudaSuccess || cudaMalloc(&dev_out, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_in, host_in.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    // Pass a 12-byte struct: n=kN, scale=3.0, offset=-1.5
    KernelParams params{kN, 3.0f, -1.5f};

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BYTES, sizeof(KernelParams)},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "struct_arg_kernel",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    void* arg_in = dev_in;
    void* arg_out = dev_out;
    void* launch_args[] = {&arg_in, &arg_out, &params};

    const dim3 block_dim(kN, 1, 1);
    const dim3 grid_dim(1, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_out.data(), dev_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    // Verify: output[i] = input[i] * scale + offset = i * 3.0 - 1.5
    for (std::uint32_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>(i) * params.scale + params.offset;
        if (std::fabs(host_out[i] - expected) > 1e-5f) {
            std::fprintf(stderr,
                         "FAIL: output[%u] = %f expected %f\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    cudaFree(dev_in);
    cudaFree(dev_out);

    std::printf("PASS: struct-by-value kernel argument (n=%u, scale=%.1f, offset=%.1f)\n",
                params.n,
                static_cast<double>(params.scale),
                static_cast<double>(params.offset));
    return 0;
}
