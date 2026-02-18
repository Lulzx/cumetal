#include "cuda.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace {

struct CallbackState {
    std::mutex mutex;
    std::condition_variable cv;
    bool called = false;
    CUresult status = CUDA_ERROR_UNKNOWN;
    CUstream stream = nullptr;
};

void driver_stream_callback(CUstream stream, CUresult status, void* user_data) {
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
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamCreate failed\n");
        return 1;
    }

    CallbackState stream_state;
    if (cuStreamAddCallback(stream, driver_stream_callback, &stream_state, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamAddCallback(stream, ...) failed\n");
        return 1;
    }

    if (!wait_for_callback(&stream_state)) {
        std::fprintf(stderr, "FAIL: stream callback was not invoked\n");
        return 1;
    }

    if (stream_state.status != CUDA_SUCCESS || stream_state.stream != stream) {
        std::fprintf(stderr, "FAIL: stream callback reported unexpected status/stream\n");
        return 1;
    }

    CallbackState default_state;
    if (cuStreamAddCallback(nullptr, driver_stream_callback, &default_state, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamAddCallback(default stream, ...) failed\n");
        return 1;
    }

    if (!wait_for_callback(&default_state)) {
        std::fprintf(stderr, "FAIL: default-stream callback was not invoked\n");
        return 1;
    }

    if (default_state.status != CUDA_SUCCESS || default_state.stream != nullptr) {
        std::fprintf(stderr, "FAIL: default-stream callback reported unexpected status/stream\n");
        return 1;
    }

    if (cuStreamAddCallback(stream, nullptr, &stream_state, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null callback should be rejected\n");
        return 1;
    }

    if (cuStreamAddCallback(stream, driver_stream_callback, &stream_state, 1) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: non-zero flags should be rejected\n");
        return 1;
    }

    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver stream callback API behaves correctly\n");
    return 0;
}
