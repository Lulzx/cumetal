#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary2(const void* fat_cubin, ...);
void __cudaRegisterFatBinaryEnd(void** fat_cubin_handle);
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
void __cudaRegisterManagedVar(void** fat_cubin_handle,
                              void** host_var_ptr_address,
                              char* device_address,
                              const char* device_name,
                              int ext,
                              std::size_t size,
                              int constant,
                              int global);
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
int g_managed_symbol = 7;

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

    CumetalFatbinImage fatbin{.metallib_path = metallib_path.c_str()};
    void** fatbin_handle = __cudaRegisterFatBinary2(&fatbin, 0, nullptr, nullptr);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary2 returned null\n");
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
    void* managed_symbol_addr = static_cast<void*>(&g_managed_symbol);
    __cudaRegisterManagedVar(fatbin_handle,
                             &managed_symbol_addr,
                             nullptr,
                             "g_managed_symbol",
                             0,
                             sizeof(g_managed_symbol),
                             0,
                             1);
    __cudaRegisterFatBinaryEnd(fatbin_handle);

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 13) % 43) * 0.25f;
        host_b[i] = static_cast<float>((i * 17) % 47) * 0.5f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c, nullptr};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock),
                        1,
                        1);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         launch_args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through fatbinary2 registration path failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
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

    __cudaUnregisterFatBinary(fatbin_handle);

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: fatbinary2/fatbinaryEnd registration symbols work for kernel launch\n");
    return 0;
}
