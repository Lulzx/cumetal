#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
cudaError_t __cudaPushCallConfiguration(dim3 grid_dim,
                                        dim3 block_dim,
                                        std::size_t shared_mem,
                                        cudaStream_t stream);
cudaError_t __cudaPopCallConfiguration(dim3* grid_dim,
                                       dim3* block_dim,
                                       std::size_t* shared_mem,
                                       void** stream);
}

namespace {

constexpr std::size_t kElementCount = 1u << 13;
constexpr std::size_t kThreadsPerBlock = 256;
constexpr std::uint32_t kCumetalFatbinMagic = 0x4C544D43u;  // "CMTL"
constexpr std::uint32_t kCumetalFatbinVersion = 1u;

struct CumetalFatbinImage {
    std::uint32_t magic = kCumetalFatbinMagic;
    std::uint32_t version = kCumetalFatbinVersion;
    const char* metallib_path = nullptr;
};

void vector_add_host_stub() {}

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

bool verify_launch(const void* func, cudaStream_t stream_handle) {
    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>(i) * 0.25f;
        host_b[i] = static_cast<float>(i % 13) * 1.75f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return false;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return false;
    }

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c, nullptr};

    dim3 grid_dim{};
    dim3 block_dim{};
    std::size_t shared_mem = 0;
    void* stream_ptr = nullptr;
    if (__cudaPopCallConfiguration(&grid_dim, &block_dim, &shared_mem, &stream_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: __cudaPopCallConfiguration failed\n");
        return false;
    }

    if (cudaLaunchKernel(func,
                         grid_dim,
                         block_dim,
                         launch_args,
                         shared_mem,
                         reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through call-config path failed\n");
        return false;
    }

    if (cudaStreamSynchronize(stream_handle) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize failed\n");
        return false;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return false;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return false;
        }
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return false;
    }

    return true;
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

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    CumetalFatbinImage fatbin{.metallib_path = metallib_path.c_str()};
    void** fatbin_handle = __cudaRegisterFatBinary(&fatbin);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           nullptr,
                           "vector_add",
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock),
                        1,
                        1);
    if (__cudaPushCallConfiguration(grid_dim, block_dim, 0, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: __cudaPushCallConfiguration failed\n");
        return 1;
    }

    if (!verify_launch(reinterpret_cast<const void*>(&vector_add_host_stub), stream)) {
        return 1;
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    if (__cudaPushCallConfiguration(grid_dim, block_dim, 0, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: second __cudaPushCallConfiguration failed\n");
        return 1;
    }

    dim3 popped_grid{};
    dim3 popped_block{};
    std::size_t popped_shared = 0;
    void* popped_stream = nullptr;
    if (__cudaPopCallConfiguration(&popped_grid, &popped_block, &popped_shared, &popped_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: second __cudaPopCallConfiguration failed\n");
        return 1;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    if (cudaMalloc(&dev_a, sizeof(float)) != cudaSuccess || cudaMalloc(&dev_b, sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_c, sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: post-unregister cudaMalloc failed\n");
        return 1;
    }

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* args[] = {&arg_a, &arg_b, &arg_c, nullptr};
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         popped_grid,
                         popped_block,
                         args,
                         popped_shared,
                         reinterpret_cast<cudaStream_t>(popped_stream)) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: unregistered host function should fail launch\n");
        return 1;
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess || cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: post-unregister cudaFree failed\n");
        return 1;
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: call configuration + registration launch path validated\n");
    return 0;
}
