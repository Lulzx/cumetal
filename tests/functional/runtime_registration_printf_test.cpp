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

// Exercises the registration-path device printf (spec §5.3, §6.5 step 10).
// The kernel squares each element and prints "idx=%u\n" via vprintf; the
// runtime auto-injects the ring-buffer arguments.  The test verifies that:
//   (a) the launch succeeds, and
//   (b) the computation is correct (output[i] == input[i] * input[i]).
// Printf output goes to stderr and is not checked by this test.

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

constexpr std::size_t kElementCount = 64;
constexpr std::size_t kThreadsPerBlock = 32;
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

void printf_scale_host_stub() {}

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
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

    // Build a minimal fatbin blob wrapping the PTX bytes.
    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx_file_bytes.size(), 0);
    FatbinBlobHeader header{};
    header.fat_size = static_cast<std::uint64_t>(ptx_file_bytes.size());
    std::memcpy(fatbin_blob.data(), &header, sizeof(header));
    std::memcpy(fatbin_blob.data() + sizeof(header), ptx_file_bytes.data(), ptx_file_bytes.size());

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

    char device_function[] = "printf_scale";
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&printf_scale_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    std::vector<float> host_in(kElementCount);
    std::vector<float> host_out(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_in[i] = static_cast<float>(i) * 0.5f + 1.0f;
    }

    void* dev_in = nullptr;
    void* dev_out = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);

    if (cudaMalloc(&dev_in, bytes) != cudaSuccess || cudaMalloc(&dev_out, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_in, host_in.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    void* arg_in = dev_in;
    void* arg_out = dev_out;
    void* args[] = {&arg_in, &arg_out, nullptr};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(
        static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock),
        1,
        1);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&printf_scale_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
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

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_in[i] * host_in[i];
        if (!nearly_equal(host_out[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    cudaFree(dev_in);
    cudaFree(dev_out);

    std::printf("PASS: registration-path device printf — launch succeeded, output correct\n");
    return 0;
}
