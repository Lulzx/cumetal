#include "cuda_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kSymbolSize = 128;
unsigned char g_symbol_bytes[kSymbolSize] = {};

bool bytes_equal(const std::vector<unsigned char>& lhs, const std::vector<unsigned char>& rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    std::vector<unsigned char> host_src(kSymbolSize);
    std::vector<unsigned char> host_out(kSymbolSize, 0);
    for (std::size_t i = 0; i < kSymbolSize; ++i) {
        host_src[i] = static_cast<unsigned char>((i * 17u + 9u) & 0xFFu);
    }

    std::fill(std::begin(g_symbol_bytes), std::end(g_symbol_bytes), 0);
    if (cudaMemcpyToSymbol(g_symbol_bytes, host_src.data(), kSymbolSize - 16, 8, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbol host->device failed\n");
        return 1;
    }
    if (cudaMemcpyFromSymbol(host_out.data(), g_symbol_bytes, kSymbolSize - 16, 8, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbol device->host failed\n");
        return 1;
    }

    std::vector<unsigned char> expected(host_src.begin(), host_src.begin() + (kSymbolSize - 16));
    std::vector<unsigned char> observed(host_out.begin(), host_out.begin() + (kSymbolSize - 16));
    if (!bytes_equal(observed, expected)) {
        std::fprintf(stderr, "FAIL: symbol round-trip mismatch\n");
        return 1;
    }

    if (cudaMemcpyToSymbol(g_symbol_bytes, host_src.data(), kSymbolSize, 0, cudaMemcpyDefault) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbol default kind failed\n");
        return 1;
    }
    std::fill(host_out.begin(), host_out.end(), 0);
    if (cudaMemcpyFromSymbol(host_out.data(), g_symbol_bytes, kSymbolSize, 0, cudaMemcpyDefault) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbol default kind failed\n");
        return 1;
    }
    if (!bytes_equal(host_out, host_src)) {
        std::fprintf(stderr, "FAIL: default-kind symbol copy mismatch\n");
        return 1;
    }

    void* device_bytes = nullptr;
    if (cudaMalloc(&device_bytes, kSymbolSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpyToSymbol(g_symbol_bytes, host_src.data(), kSymbolSize, 0, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbol before DtoD failed\n");
        return 1;
    }
    if (cudaMemcpyFromSymbol(device_bytes, g_symbol_bytes, kSymbolSize, 0, cudaMemcpyDeviceToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbol DtoD failed\n");
        return 1;
    }
    std::fill(host_out.begin(), host_out.end(), 0);
    if (cudaMemcpy(host_out.data(), device_bytes, kSymbolSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host after DtoD failed\n");
        return 1;
    }
    if (!bytes_equal(host_out, host_src)) {
        std::fprintf(stderr, "FAIL: DtoD symbol copy mismatch\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    std::vector<unsigned char> async_src(kSymbolSize);
    std::vector<unsigned char> async_out(kSymbolSize, 0);
    for (std::size_t i = 0; i < kSymbolSize; ++i) {
        async_src[i] = static_cast<unsigned char>((i * 5u + 3u) & 0xFFu);
    }

    if (cudaMemcpyToSymbolAsync(g_symbol_bytes,
                                async_src.data(),
                                kSymbolSize - 12,
                                6,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToSymbolAsync failed\n");
        return 1;
    }
    if (cudaMemcpyFromSymbolAsync(async_out.data(),
                                  g_symbol_bytes,
                                  kSymbolSize - 12,
                                  6,
                                  cudaMemcpyDeviceToHost,
                                  stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbolAsync failed\n");
        return 1;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize failed\n");
        return 1;
    }
    std::vector<unsigned char> expected_async(async_src.begin(), async_src.begin() + (kSymbolSize - 12));
    std::vector<unsigned char> observed_async(async_out.begin(), async_out.begin() + (kSymbolSize - 12));
    if (!bytes_equal(observed_async, expected_async)) {
        std::fprintf(stderr, "FAIL: async symbol copy mismatch\n");
        return 1;
    }

    if (cudaMemcpyToSymbol(g_symbol_bytes, host_src.data(), 16, 0, cudaMemcpyDeviceToHost) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue for ToSymbol DtoH kind\n");
        return 1;
    }
    if (cudaMemcpyFromSymbol(host_out.data(), g_symbol_bytes, 16, 0, cudaMemcpyHostToDevice) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue for FromSymbol HtoD kind\n");
        return 1;
    }
    if (cudaMemcpyToSymbol(g_symbol_bytes, host_src.data(), 16, 0, cudaMemcpyDeviceToDevice) !=
        cudaErrorInvalidDevicePointer) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidDevicePointer for ToSymbol DtoD host src\n");
        return 1;
    }
    if (cudaMemcpyFromSymbol(host_out.data(), g_symbol_bytes, 16, 0, cudaMemcpyDeviceToDevice) !=
        cudaErrorInvalidDevicePointer) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidDevicePointer for FromSymbol DtoD host dst\n");
        return 1;
    }
    if (cudaMemcpyToSymbol(nullptr, host_src.data(), 4, 0, cudaMemcpyHostToDevice) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue for null symbol\n");
        return 1;
    }
    if (cudaMemcpyFromSymbol(host_out.data(), nullptr, 4, 0, cudaMemcpyDeviceToHost) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue for null symbol source\n");
        return 1;
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }
    if (cudaFree(device_bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: cudaMemcpy{To,From}Symbol sync/async semantics validated\n");
    return 0;
}
