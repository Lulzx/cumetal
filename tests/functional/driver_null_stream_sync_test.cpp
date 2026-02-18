#include "cuda.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1u << 18;
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

    CUfunction vector_add = nullptr;
    if (cuModuleGetFunction(&vector_add, module, "vector_add") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleGetFunction failed\n");
        return 1;
    }

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_out(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 7) % 37) * 0.25f;
        host_b[i] = static_cast<float>((i * 11) % 23) * 1.0f;
    }

    const std::size_t bytes = kElementCount * sizeof(float);
    CUdeviceptr dev_a = 0;
    CUdeviceptr dev_b = 0;
    CUdeviceptr dev_tmp = 0;
    CUdeviceptr dev_out = 0;
    if (cuMemAlloc(&dev_a, bytes) != CUDA_SUCCESS || cuMemAlloc(&dev_b, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_tmp, bytes) != CUDA_SUCCESS || cuMemAlloc(&dev_out, bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    if (cuMemcpyHtoD(dev_a, host_a.data(), bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(dev_b, host_b.data(), bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyHtoD failed\n");
        return 1;
    }

    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cuStreamCreate failed\n");
        return 1;
    }

    CUdeviceptr s1_a = dev_a;
    CUdeviceptr s1_b = dev_b;
    CUdeviceptr s1_tmp = dev_tmp;
    void* stage1_params[] = {&s1_a, &s1_b, &s1_tmp, nullptr};

    const unsigned int grid_x =
        static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock);

    if (cuLaunchKernel(vector_add,
                       grid_x,
                       1,
                       1,
                       static_cast<unsigned int>(kThreadsPerBlock),
                       1,
                       1,
                       0,
                       stream,
                       stage1_params,
                       nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: stage1 launch on non-default stream failed\n");
        return 1;
    }

    // No explicit stream/event synchronization. Null-stream launch should fence prior stream work.
    CUdeviceptr s2_tmp = dev_tmp;
    CUdeviceptr s2_b = dev_b;
    CUdeviceptr s2_out = dev_out;
    void* stage2_params[] = {&s2_tmp, &s2_b, &s2_out, nullptr};
    if (cuLaunchKernel(vector_add,
                       grid_x,
                       1,
                       1,
                       static_cast<unsigned int>(kThreadsPerBlock),
                       1,
                       1,
                       0,
                       nullptr,
                       stage2_params,
                       nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: stage2 launch on null stream failed\n");
        return 1;
    }

    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize failed\n");
        return 1;
    }

    if (cuMemcpyDtoH(host_out.data(), dev_out, bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i] + host_b[i];
        if (!nearly_equal(host_out[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy failed\n");
        return 1;
    }
    if (cuMemFree(dev_a) != CUDA_SUCCESS || cuMemFree(dev_b) != CUDA_SUCCESS ||
        cuMemFree(dev_tmp) != CUDA_SUCCESS || cuMemFree(dev_out) != CUDA_SUCCESS) {
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

    std::printf("PASS: driver null stream waits for prior non-default stream work\n");
    return 0;
}
