#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1u << 14;
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
    std::vector<float> host_c(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 13) % 37) * 0.25f;
        host_b[i] = static_cast<float>((i * 7) % 29) * 0.5f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    if (cudaMalloc(&dev_a, kElementCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_b, kElementCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_c, kElementCount * sizeof(float)) != cudaSuccess) {
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

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    if (cudaStreamQuery(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: empty stream should query as ready\n");
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

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock),
                        1,
                        1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    const cudaError_t first_query = cudaStreamQuery(stream);
    if (first_query != cudaSuccess && first_query != cudaErrorNotReady) {
        std::fprintf(stderr, "FAIL: unexpected cudaStreamQuery status %d\n", first_query);
        return 1;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize failed\n");
        return 1;
    }

    if (cudaStreamQuery(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stream should be ready after synchronize\n");
        return 1;
    }

    if (cudaStreamQuery(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: default stream query should be ready\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, kElementCount * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: cudaStreamQuery reports stream readiness correctly\n");
    return 0;
}
