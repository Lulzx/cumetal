#include "cuda_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterVar(void** fat_cubin_handle,
                       char* host_var,
                       char* device_address,
                       const char* device_name,
                       int ext,
                       std::size_t size,
                       int constant,
                       int global);
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

constexpr std::size_t kSymbolSize = 64;
unsigned char g_host_symbol[kSymbolSize] = {};
unsigned char g_device_symbol[kSymbolSize] = {};
int g_managed_host_symbol = 0;
int g_managed_device_symbol = 0;

bool bytes_equal(const std::vector<unsigned char>& lhs, const std::vector<unsigned char>& rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    void** fatbin_handle = __cudaRegisterFatBinary(nullptr);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary returned null\n");
        return 1;
    }

    std::fill(std::begin(g_host_symbol), std::end(g_host_symbol), 0);
    std::fill(std::begin(g_device_symbol), std::end(g_device_symbol), 0);
    g_managed_host_symbol = 0;
    g_managed_device_symbol = 0;

    __cudaRegisterVar(fatbin_handle,
                      reinterpret_cast<char*>(g_host_symbol),
                      reinterpret_cast<char*>(g_device_symbol),
                      "g_symbol",
                      0,
                      kSymbolSize,
                      0,
                      1);

    void* managed_host_symbol_ptr = static_cast<void*>(&g_managed_host_symbol);
    __cudaRegisterManagedVar(fatbin_handle,
                             &managed_host_symbol_ptr,
                             reinterpret_cast<char*>(&g_managed_device_symbol),
                             "g_managed_symbol",
                             0,
                             sizeof(g_managed_device_symbol),
                             0,
                             1);

    std::vector<unsigned char> host_src(kSymbolSize);
    for (std::size_t i = 0; i < kSymbolSize; ++i) {
        host_src[i] = static_cast<unsigned char>((i * 9u + 7u) & 0xFFu);
    }

    if (cudaMemcpyToSymbol(g_host_symbol, host_src.data(), host_src.size(), 0, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbol with registered var failed\n");
        return 1;
    }

    std::vector<unsigned char> host_symbol_bytes(std::begin(g_host_symbol), std::end(g_host_symbol));
    std::vector<unsigned char> device_symbol_bytes(std::begin(g_device_symbol), std::end(g_device_symbol));
    std::vector<unsigned char> zeros(kSymbolSize, 0);
    if (!bytes_equal(host_symbol_bytes, zeros)) {
        std::fprintf(stderr, "FAIL: host symbol should not be written when var is registered\n");
        return 1;
    }
    if (!bytes_equal(device_symbol_bytes, host_src)) {
        std::fprintf(stderr, "FAIL: registered device symbol bytes mismatch\n");
        return 1;
    }

    std::vector<unsigned char> host_out(kSymbolSize, 0);
    if (cudaMemcpyFromSymbol(host_out.data(), g_host_symbol, host_out.size(), 0, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbol with registered var failed\n");
        return 1;
    }
    if (!bytes_equal(host_out, host_src)) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbol observed bytes mismatch\n");
        return 1;
    }

    if (cudaMemcpyToSymbol(g_host_symbol, host_src.data(), 8, kSymbolSize - 4, cudaMemcpyHostToDevice) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue on registered var overflow\n");
        return 1;
    }

    const int managed_value = 1234;
    if (cudaMemcpyToSymbol(&g_managed_host_symbol,
                           &managed_value,
                           sizeof(managed_value),
                           0,
                           cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbol for registered managed var failed\n");
        return 1;
    }
    if (g_managed_host_symbol != 0 || g_managed_device_symbol != managed_value) {
        std::fprintf(stderr, "FAIL: managed var registration mapping did not apply\n");
        return 1;
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    std::vector<unsigned char> fallback_src(kSymbolSize);
    for (std::size_t i = 0; i < kSymbolSize; ++i) {
        fallback_src[i] = static_cast<unsigned char>((i * 5u + 3u) & 0xFFu);
    }
    if (cudaMemcpyToSymbol(g_host_symbol, fallback_src.data(), fallback_src.size(), 0, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbol after unregister failed\n");
        return 1;
    }

    host_symbol_bytes.assign(std::begin(g_host_symbol), std::end(g_host_symbol));
    if (!bytes_equal(host_symbol_bytes, fallback_src)) {
        std::fprintf(stderr, "FAIL: host symbol fallback path mismatch after unregister\n");
        return 1;
    }

    std::printf("PASS: __cudaRegisterVar/__cudaRegisterManagedVar symbol mapping validated\n");
    return 0;
}
