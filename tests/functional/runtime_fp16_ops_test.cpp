#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Test: FP16 arithmetic in Metal (half-precision add).
//
// Kernel: val = half(input[i]); result = val + half(1.0); output[i] = float(result)
// For integer inputs in [0, 255]:
//   - half(i) is exactly representable (half mantissa covers integers up to 2048)
//   - half(i) + half(1.0) = half(i + 1), also exactly representable
//   - So expected output[i] = float(i + 1)

namespace {

constexpr uint32_t kN         = 256;
constexpr uint32_t kBlockSize = 256;

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

    float* d_input  = nullptr;
    float* d_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_input),  kN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_output), kN * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    std::vector<float> h_input(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    if (cudaMemcpy(d_input, h_input.data(), kN * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H->D failed\n");
        return 1;
    }
    if (cudaMemset(d_output, 0, kN * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "fp16_add",
        .arg_count     = 2,
        .arg_info      = kArgInfo,
    };

    void* arg_input  = d_input;
    void* arg_output = d_output;
    void* args[]     = {&arg_input, &arg_output};

    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(kN / kBlockSize, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::vector<float> h_output(kN, -1.0f);
    if (cudaMemcpy(h_output.data(), d_output, kN * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>(i + 1);
        if (h_output[i] != expected) {
            std::fprintf(stderr, "FAIL: output[%u] = %.6f, expected %.6f\n", i, h_output[i],
                         expected);
            return 1;
        }
    }

    if (cudaFree(d_input) != cudaSuccess || cudaFree(d_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: FP16 arithmetic correctness validated (%u elements)\n", kN);
    return 0;
}
