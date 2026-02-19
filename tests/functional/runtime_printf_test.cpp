#include "cuda_runtime.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests device printf via a ring buffer (spec §5.3).
// The kernel writes [fmt_id][n_args][arg0][arg1] records to a shared buffer; the CPU
// reads and formats them after kernel completion.  Buffer overflow (capacity < records
// needed) is tested separately with a small capacity to verify silent drop behavior.

namespace {

constexpr std::size_t kN = 8;
// The format string table maps format IDs to printf format strings.
// Format 0 is "thread %u: %.2f\n" (2 args: tid:u32, result:float-bits-as-u32).
const char* kFormatTable[] = {"thread %u: %.2f\n"};

// Drain the ring buffer: read records and print them.  Returns number of records
// that were successfully printed (not overflowed).
int drain_printf_buf(const std::uint32_t* words, std::uint32_t capacity_words) {
    const std::uint32_t write_count = words[0];  // words consumed (not counting buf[0])
    int printed = 0;
    std::uint32_t pos = 0;
    while (pos + 3 <= write_count) {  // need at least fmt_id + n_args + 1 arg
        const std::uint32_t fmt_id = words[pos + 1];
        const std::uint32_t n_args = words[pos + 2];
        if (pos + 2 + n_args > write_count) break;
        if (pos + 2 + n_args >= capacity_words) {
            // This record was overflowed (partially written); stop.
            break;
        }
        if (fmt_id < sizeof(kFormatTable) / sizeof(kFormatTable[0])) {
            if (n_args == 2) {
                const std::uint32_t a0 = words[pos + 3];
                const std::uint32_t a1 = words[pos + 4];
                float f1 = std::bit_cast<float>(a1);
                std::fprintf(stderr, kFormatTable[fmt_id], a0, static_cast<double>(f1));
                ++printed;
            }
        }
        pos += 2 + n_args;
    }
    return printed;
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

    std::vector<float> host_in(kN);
    std::vector<float> host_out(kN, 0.0f);
    for (std::size_t i = 0; i < kN; ++i) {
        host_in[i] = static_cast<float>(i) * 1.0f;
    }

    void* dev_in = nullptr;
    void* dev_out = nullptr;
    const std::size_t bytes = kN * sizeof(float);

    if (cudaMalloc(&dev_in, bytes) != cudaSuccess || cudaMalloc(&dev_out, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_in, host_in.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy input host->device failed\n");
        return 1;
    }

    // ── Test 1: normal run — buffer large enough to hold all records ──────────
    // Each record is 4 words (fmt_id + n_args + 2 args).  kN threads → kN*4 words.
    // Add 1 for the atomic counter at buf[0].
    const std::uint32_t kBufCapWords = static_cast<std::uint32_t>(kN) * 4 + 1;
    void* dev_printf_buf = nullptr;
    const std::size_t printf_buf_bytes = kBufCapWords * sizeof(std::uint32_t);
    if (cudaMalloc(&dev_printf_buf, printf_buf_bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc printf_buf failed\n");
        return 1;
    }
    if (cudaMemset(dev_printf_buf, 0, printf_buf_bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset printf_buf failed\n");
        return 1;
    }

    void* dev_cap = nullptr;
    if (cudaMalloc(&dev_cap, sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc cap failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_cap, &kBufCapWords, sizeof(kBufCapWords), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy cap failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "printf_kernel",
        .arg_count = 4,
        .arg_info = kArgInfo,
    };

    void* arg_in = dev_in;
    void* arg_out = dev_out;
    void* arg_pbuf = dev_printf_buf;
    void* arg_cap = dev_cap;
    void* launch_args[] = {&arg_in, &arg_out, &arg_pbuf, &arg_cap};

    const dim3 block_dim(kN, 1, 1);
    const dim3 grid_dim(1, 1, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_out.data(), dev_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy output device->host failed\n");
        return 1;
    }

    // Verify computation: output[i] = input[i] * 2.0 + 1.0
    for (std::size_t i = 0; i < kN; ++i) {
        const float expected = host_in[i] * 2.0f + 1.0f;
        if (std::fabs(host_out[i] - expected) > 1e-5f) {
            std::fprintf(stderr,
                         "FAIL: output[%zu] = %f expected %f\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    // Drain and verify printf buffer
    std::vector<std::uint32_t> host_buf(kBufCapWords, 0);
    if (cudaMemcpy(host_buf.data(), dev_printf_buf, printf_buf_bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy printf_buf device->host failed\n");
        return 1;
    }

    const int printed = drain_printf_buf(host_buf.data(), kBufCapWords);
    if (printed != static_cast<int>(kN)) {
        std::fprintf(stderr,
                     "FAIL: expected %zu printf records, got %d\n",
                     kN,
                     printed);
        return 1;
    }

    // ── Test 2: overflow — buffer too small to hold all records ──────────────
    // Capacity = 5 words → fits buf[0] + 1 record (4 words) only.
    const std::uint32_t kSmallCap = 5;
    void* dev_small_buf = nullptr;
    if (cudaMalloc(&dev_small_buf, kSmallCap * sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc small_buf failed\n");
        return 1;
    }
    if (cudaMemset(dev_small_buf, 0, kSmallCap * sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset small_buf failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_cap, &kSmallCap, sizeof(kSmallCap), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy small cap failed\n");
        return 1;
    }

    void* arg_sbuf = dev_small_buf;
    void* overflow_args[] = {&arg_in, &arg_out, &arg_sbuf, &arg_cap};

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, overflow_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: overflow cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: overflow cudaDeviceSynchronize failed\n");
        return 1;
    }

    // With capacity=5, at most 1 record fits (pos=0 → pos+4=4 < 5).
    // Remaining records are silently dropped.
    std::vector<std::uint32_t> small_host_buf(kSmallCap, 0);
    if (cudaMemcpy(small_host_buf.data(),
                   dev_small_buf,
                   kSmallCap * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy small_buf device->host failed\n");
        return 1;
    }

    // The write_count will exceed capacity (threads still atomically added their slice),
    // but only the records that fit within [1..kSmallCap-1] contain valid data.
    // drain_printf_buf stops at the capacity boundary.
    const int overflow_printed = drain_printf_buf(small_host_buf.data(), kSmallCap);
    if (overflow_printed < 1) {
        std::fprintf(stderr,
                     "FAIL: expected at least 1 printf record in overflow test, got %d\n",
                     overflow_printed);
        return 1;
    }
    if (overflow_printed >= static_cast<int>(kN)) {
        std::fprintf(stderr,
                     "FAIL: overflow test should have dropped some records, but printed=%d for "
                     "capacity=%u\n",
                     overflow_printed,
                     kSmallCap);
        return 1;
    }

    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_printf_buf);
    cudaFree(dev_small_buf);
    cudaFree(dev_cap);

    std::printf("PASS: device printf ring buffer — %d records printed, overflow drop verified\n",
                printed);
    return 0;
}
