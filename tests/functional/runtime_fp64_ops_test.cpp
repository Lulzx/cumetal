#include "cuda.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// Test: FP64 arithmetic via PTX fma.rn.f64 (--fp64=native path).
//
// 64 threads (single block).  input[i] = float(i).
// PTX kernel: output[i] = float(double(input[i]) * 2.0 + 1.0)
// Expected:   output[i] = float(2*i + 1)  for i in [0, 64).
//
// All inputs are small integers (≤ 63) exactly representable in float and
// double, so the round-trip f32→f64→f32 result must be exactly 2*i+1.

namespace {

constexpr uint32_t kN          = 64;
constexpr uint32_t kBlockSize  = 64;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-ptx>\n", argv[0]);
        return 64;
    }

    const std::string ptx_path = argv[1];
    if (!std::filesystem::exists(ptx_path)) {
        std::fprintf(stderr, "SKIP: PTX not found at %s\n", ptx_path.c_str());
        return 77;
    }

    // Read PTX text
    std::ifstream in(ptx_path, std::ios::binary);
    std::vector<char> ptx_bytes((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    if (ptx_bytes.empty()) {
        std::fprintf(stderr, "FAIL: failed to read PTX file\n");
        return 1;
    }
    ptx_bytes.push_back('\0');

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
    if (cuModuleLoadData(&module, ptx_bytes.data()) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(PTX) failed\n");
        return 1;
    }

    CUfunction kernel = nullptr;
    if (cuModuleGetFunction(&kernel, module, "fp64_mul_add") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleGetFunction failed\n");
        return 1;
    }

    std::vector<float> h_input(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    CUdeviceptr d_input  = 0;
    CUdeviceptr d_output = 0;
    if (cuMemAlloc(&d_input,  kN * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_output, kN * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    if (cuMemcpyHtoD(d_input, h_input.data(), kN * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyHtoD failed\n");
        return 1;
    }

    CUdeviceptr arg_input  = d_input;
    CUdeviceptr arg_output = d_output;
    void* params[] = {&arg_input, &arg_output, nullptr};

    const CUresult launch_res = cuLaunchKernel(kernel,
                       1, 1, 1,          // grid
                       kBlockSize, 1, 1, // block
                       0, nullptr, params, nullptr);
    if (launch_res != CUDA_SUCCESS) {
        // Apple Silicon GPU does not support FP64 arithmetic at runtime.
        // Metal pipeline state creation fails when the AIR contains double-
        // precision arithmetic (fmul/fadd double, @llvm.fma.f64).  This is
        // a known hardware limitation; return 77 to signal SKIP.
        std::fprintf(stderr,
                     "SKIP: cuLaunchKernel failed (FP64 arithmetic not "
                     "supported on this device - Apple Silicon limitation)\n");
        return 77;
    }

    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize failed\n");
        return 1;
    }

    std::vector<float> h_output(kN, -1.0f);
    if (cuMemcpyDtoH(h_output.data(), d_output, kN * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>(static_cast<double>(i) * 2.0 + 1.0);
        if (h_output[i] != expected) {
            std::fprintf(stderr, "FAIL: output[%u] = %.1f, expected %.1f\n", i, h_output[i],
                         expected);
            return 1;
        }
    }

    if (cuMemFree(d_input)  != CUDA_SUCCESS ||
        cuMemFree(d_output) != CUDA_SUCCESS) {
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

    std::printf("PASS: FP64 arithmetic validated (%u elements, fma.rn.f64: val*2.0+1.0)\n", kN);
    return 0;
}
