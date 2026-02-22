#include "cuda.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

struct HostFuncState {
    std::atomic<int> call_count{0};
    std::atomic<void*> last_user_data{nullptr};
};

void host_func(void* user_data) {
    auto* state = static_cast<HostFuncState*>(user_data);
    if (state) {
        state->last_user_data.store(user_data);
        state->call_count.fetch_add(1);
    }
}

bool wait_for_calls(HostFuncState* state, int expected, int timeout_ms = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (state->call_count.load() < expected) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

}  // namespace

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }
    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    // Test 1: null fn is rejected.
    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamCreate failed\n");
        return 1;
    }
    if (cuLaunchHostFunc(stream, nullptr, nullptr) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null fn should be rejected\n");
        return 1;
    }

    // Test 2: fn is called on a user stream.
    HostFuncState state;
    if (cuLaunchHostFunc(stream, host_func, &state) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuLaunchHostFunc(stream, ...) failed\n");
        return 1;
    }
    if (!wait_for_calls(&state, 1)) {
        std::fprintf(stderr, "FAIL: host func was not called on user stream\n");
        return 1;
    }
    if (state.last_user_data.load() != &state) {
        std::fprintf(stderr, "FAIL: user_data was not passed correctly\n");
        return 1;
    }

    // Test 3: fn is called on the default stream (nullptr).
    HostFuncState default_state;
    if (cuLaunchHostFunc(nullptr, host_func, &default_state) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuLaunchHostFunc(default stream, ...) failed\n");
        return 1;
    }
    if (!wait_for_calls(&default_state, 1)) {
        std::fprintf(stderr, "FAIL: host func was not called on default stream\n");
        return 1;
    }

    // Test 4: multiple successive calls are all invoked.
    HostFuncState multi_state;
    for (int i = 0; i < 3; ++i) {
        if (cuLaunchHostFunc(stream, host_func, &multi_state) != CUDA_SUCCESS) {
            std::fprintf(stderr, "FAIL: cuLaunchHostFunc call %d failed\n", i);
            return 1;
        }
    }
    if (!wait_for_calls(&multi_state, 3)) {
        std::fprintf(stderr, "FAIL: expected 3 host func calls, got %d\n", multi_state.call_count.load());
        return 1;
    }

    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy failed\n");
        return 1;
    }
    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuLaunchHostFunc API behaves correctly\n");
    return 0;
}
