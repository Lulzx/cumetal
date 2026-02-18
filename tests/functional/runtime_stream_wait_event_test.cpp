#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1u << 15;
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

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_out(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 9) % 41) * 0.25f;
        host_b[i] = static_cast<float>((i * 7) % 29) * 1.5f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_tmp = nullptr;
    void* dev_out = nullptr;

    if (cudaMalloc(&dev_a, kElementCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_b, kElementCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_tmp, kElementCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_out, kElementCount * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a.data(), kElementCount * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), kElementCount * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;
    if (cudaStreamCreate(&stream1) != cudaSuccess || cudaStreamCreate(&stream2) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    cudaEvent_t stage1_done = nullptr;
    if (cudaEventCreate(&stage1_done) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaEventCreate failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "vector_add",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock),
                        1,
                        1);

    void* stage1_a = dev_a;
    void* stage1_b = dev_b;
    void* stage1_tmp = dev_tmp;
    void* stage1_args[] = {&stage1_a, &stage1_b, &stage1_tmp};

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, stage1_args, 0, stream1) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stage1 cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaEventRecord(stage1_done, stream1) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaEventRecord(stage1_done) failed\n");
        return 1;
    }

    if (cudaStreamWaitEvent(stream2, stage1_done, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamWaitEvent failed\n");
        return 1;
    }

    if (cudaEventQuery(stage1_done) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stage1_done should be complete after stream wait\n");
        return 1;
    }

    void* stage2_tmp = dev_tmp;
    void* stage2_b = dev_b;
    void* stage2_out = dev_out;
    void* stage2_args[] = {&stage2_tmp, &stage2_b, &stage2_out};

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, stage2_args, 0, stream2) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stage2 cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaStreamSynchronize(stream2) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize(stream2) failed\n");
        return 1;
    }

    if (cudaMemcpy(host_out.data(), dev_out, kElementCount * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + 2.0f * host_b[i];
        if (!nearly_equal(host_out[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    if (cudaEventDestroy(stage1_done) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaEventDestroy failed\n");
        return 1;
    }

    if (cudaStreamDestroy(stream1) != cudaSuccess || cudaStreamDestroy(stream2) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_tmp) != cudaSuccess || cudaFree(dev_out) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: cudaStreamWaitEvent enforced cross-stream ordering\n");
    return 0;
}
