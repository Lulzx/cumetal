#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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
}

namespace {

constexpr std::size_t kElementCount = 1u << 13;
constexpr std::size_t kThreadsPerBlock = 256;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;

struct FatbinWrapper {
    std::uint32_t magic = kFatbinWrapperMagic;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinBlobHeader {
    std::uint32_t magic = kFatbinBlobMagic;
    std::uint16_t version = 1;
    std::uint16_t header_size = 16;
    std::uint64_t fat_size = 0;
};

void vector_add_host_stub() {}

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

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

    std::ifstream ptx_in(ptx_path, std::ios::binary);
    std::vector<char> ptx_file_bytes((std::istreambuf_iterator<char>(ptx_in)),
                                     std::istreambuf_iterator<char>());
    if (ptx_file_bytes.empty()) {
        std::fprintf(stderr, "FAIL: failed to read PTX bytes\n");
        return 1;
    }

    std::vector<char> ptx_bytes = ptx_file_bytes;
    ptx_bytes.push_back('\0');

    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx_file_bytes.size(), 0);
    FatbinBlobHeader header{};
    header.fat_size = static_cast<std::uint64_t>(ptx_file_bytes.size());
    std::memcpy(fatbin_blob.data(), &header, sizeof(header));
    std::memcpy(fatbin_blob.data() + sizeof(header), ptx_file_bytes.data(), ptx_file_bytes.size());

    FatbinBlobHeader padded_header{};
    padded_header.header_size = 64;
    padded_header.fat_size = static_cast<std::uint64_t>(ptx_file_bytes.size());
    std::vector<std::uint8_t> fatbin_blob_padded(padded_header.header_size + ptx_file_bytes.size(), 0);
    std::memcpy(fatbin_blob_padded.data(), &padded_header, sizeof(padded_header));
    std::memcpy(fatbin_blob_padded.data() + padded_header.header_size,
                ptx_file_bytes.data(),
                ptx_file_bytes.size());

    FatbinWrapper wrapper{};
    wrapper.data = fatbin_blob.data();

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    void** fatbin_handle = __cudaRegisterFatBinary(&wrapper);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary returned null\n");
        return 1;
    }

    char device_function[] = "vector_add";
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 11) % 37) * 0.25f;
        host_b[i] = static_cast<float>((i * 7) % 29) * 1.5f;
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
    void* args[] = {&arg_a, &arg_b, &arg_c, nullptr};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock),
                        1,
                        1);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through fatbin PTX registration failed\n");
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

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: launch should fail after __cudaUnregisterFatBinary\n");
        return 1;
    }

    fatbin_handle = __cudaRegisterFatBinary(fatbin_blob.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (direct fatbin blob) returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through direct fatbin blob registration failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize after direct fatbin blob launch failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host after direct fatbin blob launch failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: direct fatbin blob mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    fatbin_handle = __cudaRegisterFatBinary(fatbin_blob_padded.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (padded fatbin blob) returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through padded fatbin blob registration failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize after padded fatbin blob launch failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host after padded fatbin blob launch failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: padded fatbin blob mismatch at %zu (got=%f expected=%f)\n",
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

    std::printf("PASS: runtime registration supports fatbin PTX launch path (wrapper and direct blob)\n");
    return 0;
}
