#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests the cp.async emulation path (spec §5.1.1): global→threadgroup copy via
// synchronous load+store+barrier.  The kernel loads input into a threadgroup tile,
// barriers, then writes tile[tid]*scale to output.  Correctness verifies the
// barrier-ordered read from shared memory is correct.

namespace {

constexpr std::uint32_t kBlockSize = 128;
constexpr std::uint32_t kNumBlocks = 4;
constexpr std::uint32_t kN = kBlockSize * kNumBlocks;
constexpr float kScale = 2.5f;

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
    for (std::uint32_t i = 0; i < kN; ++i) {
        host_in[i] = static_cast<float>(i) * 0.5f + 1.0f;
    }

    void* dev_in = nullptr;
    void* dev_out = nullptr;
    if (cudaMalloc(&dev_in, kN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_out, kN * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_in, host_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    std::uint32_t n_val = kN;
    float scale_val = kScale;

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BYTES, sizeof(std::uint32_t)},
        {CUMETAL_ARG_BYTES, sizeof(float)},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "cp_async_emul_kernel",
        .arg_count = 4,
        .arg_info = kArgInfo,
    };

    void* launch_args[] = {&dev_in, &dev_out, &n_val, &scale_val};

    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(kNumBlocks, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::vector<float> host_out(kN, 0.0f);
    if (cudaMemcpy(host_out.data(), dev_out, kN * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::uint32_t i = 0; i < kN; ++i) {
        const float expected = host_in[i] * kScale;
        if (std::fabs(host_out[i] - expected) > 1e-4f) {
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

    std::printf("PASS: cp.async emulation kernel (n=%u, scale=%.1f)\n", kN, static_cast<double>(kScale));
    return 0;
}
