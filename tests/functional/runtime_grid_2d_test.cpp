#include "cuda_runtime.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Test: 2D grid and block configurations.
//
// Launch a 4x4 grid of 2x2 blocks (total 64 threads).
// The kernel writes each thread's linear index into output[linear_idx].
// Expected: output[i] == i for all i in [0, 64).

namespace {

constexpr uint32_t kGridX  = 4;
constexpr uint32_t kGridY  = 4;
constexpr uint32_t kBlockX = 2;
constexpr uint32_t kBlockY = 2;
constexpr uint32_t kN      = kGridX * kGridY * kBlockX * kBlockY;

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

    uint32_t* d_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_output), kN * sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemset(d_output, 0xff, kN * sizeof(uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name   = "grid_2d",
        .arg_count     = 1,
        .arg_info      = kArgInfo,
    };

    void* arg_output = d_output;
    void* args[]     = {&arg_output};

    const dim3 block_dim(kBlockX, kBlockY, 1);
    const dim3 grid_dim(kGridX, kGridY, 1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    std::vector<uint32_t> h_output(kN, 0xffffffff);
    if (cudaMemcpy(h_output.data(), d_output, kN * sizeof(uint32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < kN; ++i) {
        if (h_output[i] != i) {
            std::fprintf(stderr, "FAIL: output[%u] = %u, expected %u\n", i, h_output[i], i);
            return 1;
        }
    }

    if (cudaFree(d_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: 2D grid indexing validated (%ux%u grid, %ux%u block, %u total threads)\n",
                kGridX, kGridY, kBlockX, kBlockY, kN);
    return 0;
}
