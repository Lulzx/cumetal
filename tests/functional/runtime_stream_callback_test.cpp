#include "cuda_runtime.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace {

struct CallbackState {
    std::mutex mutex;
    std::condition_variable cv;
    bool called = false;
    cudaError_t status = cudaErrorUnknown;
    cudaStream_t stream = nullptr;
};

void runtime_stream_callback(cudaStream_t stream, cudaError_t status, void* user_data) {
    auto* state = static_cast<CallbackState*>(user_data);
    if (state == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->called = true;
        state->status = status;
        state->stream = stream;
    }
    state->cv.notify_all();
}

bool wait_for_callback(CallbackState* state) {
    if (state == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    return state->cv.wait_for(lock, std::chrono::seconds(5), [state]() { return state->called; });
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    CallbackState stream_state;
    if (cudaStreamAddCallback(stream, runtime_stream_callback, &stream_state, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamAddCallback(stream, ...) failed\n");
        return 1;
    }

    if (!wait_for_callback(&stream_state)) {
        std::fprintf(stderr, "FAIL: stream callback was not invoked\n");
        return 1;
    }

    if (stream_state.status != cudaSuccess || stream_state.stream != stream) {
        std::fprintf(stderr, "FAIL: stream callback reported unexpected status/stream\n");
        return 1;
    }

    CallbackState default_state;
    if (cudaStreamAddCallback(nullptr, runtime_stream_callback, &default_state, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamAddCallback(default stream, ...) failed\n");
        return 1;
    }

    if (!wait_for_callback(&default_state)) {
        std::fprintf(stderr, "FAIL: default-stream callback was not invoked\n");
        return 1;
    }

    if (default_state.status != cudaSuccess || default_state.stream != nullptr) {
        std::fprintf(stderr, "FAIL: default-stream callback reported unexpected status/stream\n");
        return 1;
    }

    if (cudaStreamAddCallback(stream, nullptr, &stream_state, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null callback should be rejected\n");
        return 1;
    }

    if (cudaStreamAddCallback(stream, runtime_stream_callback, &stream_state, 1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: non-zero flags should be rejected\n");
        return 1;
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: runtime stream callback API behaves correctly\n");
    return 0;
}
