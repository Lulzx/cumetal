#include "cuda_runtime.h"

#include <cstdio>

// Tests peer-to-peer access stub APIs (spec §8: "No multi-GPU on Apple Silicon").
// All peer access is via a single device (device 0). The spec stubs:
// - cudaDeviceCanAccessPeer: returns 0 (cannot access; single GPU)
// - cudaDeviceEnablePeerAccess: returns cudaErrorInvalidValue (single GPU)
// - cudaDeviceDisablePeerAccess: returns cudaErrorInvalidValue (single GPU)

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // cudaDeviceCanAccessPeer: null canAccessPeer is rejected.
    if (cudaDeviceCanAccessPeer(nullptr, 0, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null canAccessPeer should be rejected\n");
        return 1;
    }

    // cudaDeviceCanAccessPeer: device 0 → device 0 returns 0 (no self-peer on Apple Silicon).
    int can_access = -1;
    if (cudaDeviceCanAccessPeer(&can_access, 0, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceCanAccessPeer failed\n");
        return 1;
    }
    if (can_access != 0) {
        std::fprintf(stderr, "FAIL: canAccessPeer should be 0 (single GPU), got %d\n", can_access);
        return 1;
    }

    // cudaDeviceEnablePeerAccess: returns error since there's no second device.
    cudaError_t err = cudaDeviceEnablePeerAccess(0, 0);
    if (err == cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceEnablePeerAccess should fail on single-GPU system\n");
        return 1;
    }

    // cudaDeviceDisablePeerAccess: returns error similarly.
    err = cudaDeviceDisablePeerAccess(0);
    if (err == cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceDisablePeerAccess should fail on single-GPU system\n");
        return 1;
    }

    std::printf("PASS: peer access APIs correctly report single-GPU limitations\n");
    return 0;
}
