#include "cuda_runtime.h"

#include "allocation_table.h"
#include "metal_backend.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <new>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct cudaStream_st {};
struct cudaEvent_st {
    bool disable_timing = false;
    bool recorded_once = false;
    bool complete = true;
    bool timing_valid = false;
    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    std::uint64_t ticket = 0;
    std::chrono::steady_clock::time_point timestamp{};
    std::mutex mutex;
};

namespace {

struct RuntimeState {
    std::once_flag init_once;
    cudaError_t init_status = cudaSuccess;
    std::string init_error;
    cumetal::rt::AllocationTable allocations;
    std::mutex stream_mutex;
    std::unordered_map<cudaStream_t, std::shared_ptr<cumetal::metal_backend::Stream>> streams;
};

RuntimeState& runtime_state() {
    static RuntimeState state;
    return state;
}

thread_local cudaError_t tls_last_error = cudaSuccess;

void set_last_error(cudaError_t error) {
    tls_last_error = error;
}

cudaError_t ensure_initialized() {
    RuntimeState& state = runtime_state();
    std::call_once(state.init_once, [&state]() {
        std::string error;
        state.init_status = cumetal::metal_backend::initialize(&error);
        state.init_error = error;
    });
    return state.init_status;
}

cudaError_t fail(cudaError_t error) {
    set_last_error(error);
    return error;
}

bool resolve_stream_handle(cudaStream_t stream,
                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (stream == nullptr || out_stream == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }

    *out_stream = found->second;
    return true;
}

bool erase_stream_handle(cudaStream_t stream,
                         std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (stream == nullptr || out_stream == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }

    *out_stream = std::move(found->second);
    state.streams.erase(found);
    return true;
}

cudaError_t validate_memcpy_kind(cudaMemcpyKind kind) {
    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyDefault:
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t synchronize_stream_for_host_op(cudaStream_t stream,
                                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (out_stream != nullptr) {
        out_stream->reset();
    }
    if (stream == nullptr) {
        return cudaSuccess;
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (!resolve_stream_handle(stream, &backend_stream)) {
        return cudaErrorInvalidValue;
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::stream_synchronize(backend_stream, &error);
    if (status != cudaSuccess) {
        return status;
    }

    if (out_stream != nullptr) {
        *out_stream = std::move(backend_stream);
    }
    return cudaSuccess;
}

cudaError_t update_event_completion(cudaEvent_t event, bool wait_for_completion) {
    if (event == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    std::uint64_t ticket = 0;
    {
        std::lock_guard<std::mutex> lock(event->mutex);
        if (event->complete) {
            return cudaSuccess;
        }
        stream = event->stream;
        ticket = event->ticket;
    }

    if (stream == nullptr || ticket == 0) {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->complete = true;
        if (!event->disable_timing) {
            event->timestamp = std::chrono::steady_clock::now();
            event->timing_valid = true;
        }
        return cudaSuccess;
    }

    std::string error;
    bool is_complete = false;
    if (wait_for_completion) {
        const cudaError_t status = cumetal::metal_backend::stream_wait_ticket(stream, ticket, &error);
        if (status != cudaSuccess) {
            return status;
        }
        is_complete = true;
    } else {
        const cudaError_t status = cumetal::metal_backend::stream_query_ticket(stream, ticket,
                                                                                &is_complete, &error);
        if (status != cudaSuccess) {
            return status;
        }
        if (!is_complete) {
            return cudaErrorNotReady;
        }
    }

    std::lock_guard<std::mutex> lock(event->mutex);
    event->complete = true;
    if (!event->disable_timing) {
        event->timestamp = std::chrono::steady_clock::now();
        event->timing_valid = true;
    }
    return cudaSuccess;
}

}  // namespace

extern "C" {

cudaError_t cudaInit(unsigned int flags) {
    if (flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t status = ensure_initialized();
    return fail(status);
}

cudaError_t cudaMalloc(void** dev_ptr, size_t size) {
    if (dev_ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::string error;
    const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(size, &buffer, &error);
    if (alloc_status != cudaSuccess || buffer == nullptr) {
        return fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation : alloc_status);
    }

    void* base = buffer->contents();
    if (base == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    if (!state.allocations.insert(base, size, std::move(buffer), &error)) {
        return fail(cudaErrorMemoryAllocation);
    }

    *dev_ptr = base;
    return fail(cudaSuccess);
}

cudaError_t cudaMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
    if (flags != 0) {
        return fail(cudaErrorInvalidValue);
    }
    return cudaMalloc(dev_ptr, size);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t cudaFree(void* dev_ptr) {
    if (dev_ptr == nullptr) {
        return fail(cudaSuccess);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    RuntimeState& state = runtime_state();
    if (!state.allocations.erase(dev_ptr)) {
        return fail(cudaErrorInvalidDevicePointer);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if ((dst == nullptr || src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t kind_status = validate_memcpy_kind(kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }

    if (count > 0) {
        std::memcpy(dst, src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyAsync(void* dst,
                            const void* src,
                            size_t count,
                            cudaMemcpyKind kind,
                            cudaStream_t stream) {
    if ((dst == nullptr || src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t kind_status = validate_memcpy_kind(kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemset(void* dev_ptr, int value, size_t count) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    if (count > 0) {
        std::memset(dev_ptr, value, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, cudaStream_t stream) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memset(dev_ptr, value, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::synchronize(&error);
    return fail(status);
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    if (stream == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    std::string error;
    const cudaError_t status = cumetal::metal_backend::create_stream(&backend_stream, &error);
    if (status != cudaSuccess || backend_stream == nullptr) {
        return fail(status == cudaSuccess ? cudaErrorUnknown : status);
    }

    auto* handle = new (std::nothrow) cudaStream_st{};
    if (handle == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    {
        std::lock_guard<std::mutex> lock(state.stream_mutex);
        state.streams.emplace(handle, std::move(backend_stream));
    }

    *stream = handle;
    return fail(cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (!erase_stream_handle(stream, &backend_stream)) {
        return fail(cudaErrorInvalidValue);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::destroy_stream(backend_stream, &error);
    delete stream;
    return fail(status);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (stream == nullptr) {
        return cudaDeviceSynchronize();
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (!resolve_stream_handle(stream, &backend_stream)) {
        return fail(cudaErrorInvalidValue);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::stream_synchronize(backend_stream, &error);
    return fail(status);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (stream == nullptr) {
        std::vector<std::shared_ptr<cumetal::metal_backend::Stream>> streams;
        {
            RuntimeState& state = runtime_state();
            std::lock_guard<std::mutex> lock(state.stream_mutex);
            streams.reserve(state.streams.size());
            for (const auto& it : state.streams) {
                streams.push_back(it.second);
            }
        }

        for (const auto& backend_stream : streams) {
            std::uint64_t tail_ticket = 0;
            bool complete = true;
            std::string error;
            const cudaError_t tail_status =
                cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
            if (tail_status != cudaSuccess) {
                return fail(tail_status);
            }
            const cudaError_t query_status =
                cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket,
                                                            &complete, &error);
            if (query_status != cudaSuccess) {
                return fail(query_status);
            }
            if (!complete) {
                return fail(cudaErrorNotReady);
            }
        }

        return fail(cudaSuccess);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (!resolve_stream_handle(stream, &backend_stream)) {
        return fail(cudaErrorInvalidValue);
    }

    std::uint64_t tail_ticket = 0;
    bool complete = true;
    std::string error;
    const cudaError_t tail_status =
        cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
    if (tail_status != cudaSuccess) {
        return fail(tail_status);
    }
    const cudaError_t query_status =
        cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket, &complete, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }
    return fail(complete ? cudaSuccess : cudaErrorNotReady);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    if (event == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    if (stream != nullptr) {
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        if (!resolve_stream_handle(stream, &backend_stream)) {
            return fail(cudaErrorInvalidValue);
        }
    }

    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/true);
    return fail(status);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const unsigned int unsupported_flags = flags & ~(cudaEventDefault | cudaEventBlockingSync |
                                                      cudaEventDisableTiming);
    if (unsupported_flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    auto* created = new (std::nothrow) cudaEvent_st{};
    if (created == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    created->disable_timing = (flags & cudaEventDisableTiming) != 0;
    created->complete = true;
    created->recorded_once = false;
    created->timing_valid = false;
    created->ticket = 0;
    *event = created;
    return fail(cudaSuccess);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    delete event;
    return fail(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    std::uint64_t tail_ticket = 0;
    bool complete = true;

    if (stream != nullptr) {
        if (!resolve_stream_handle(stream, &backend_stream)) {
            return fail(cudaErrorInvalidValue);
        }

        std::string error;
        const cudaError_t tail_status =
            cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
        if (tail_status != cudaSuccess) {
            return fail(tail_status);
        }

        if (tail_ticket > 0) {
            const cudaError_t query_status =
                cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket,
                                                            &complete, &error);
            if (query_status != cudaSuccess) {
                return fail(query_status);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->stream = std::move(backend_stream);
        event->ticket = tail_ticket;
        event->recorded_once = true;
        event->complete = complete;
        event->timing_valid = false;
        if (event->complete && !event->disable_timing) {
            event->timestamp = std::chrono::steady_clock::now();
            event->timing_valid = true;
        }
    }

    return fail(cudaSuccess);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/true);
    return fail(status);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/false);
    return fail(status);
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t start_status = update_event_completion(start, /*wait_for_completion=*/false);
    if (start_status != cudaSuccess) {
        return fail(start_status);
    }
    const cudaError_t end_status = update_event_completion(end, /*wait_for_completion=*/false);
    if (end_status != cudaSuccess) {
        return fail(end_status);
    }

    std::chrono::steady_clock::time_point start_timestamp;
    std::chrono::steady_clock::time_point end_timestamp;
    {
        std::lock_guard<std::mutex> lock(start->mutex);
        if (start->disable_timing || !start->recorded_once || !start->timing_valid) {
            return fail(cudaErrorInvalidValue);
        }
        start_timestamp = start->timestamp;
    }
    {
        std::lock_guard<std::mutex> lock(end->mutex);
        if (end->disable_timing || !end->recorded_once || !end->timing_valid) {
            return fail(cudaErrorInvalidValue);
        }
        end_timestamp = end->timestamp;
    }

    *ms = std::chrono::duration<float, std::milli>(end_timestamp - start_timestamp).count();
    return fail(cudaSuccess);
}

cudaError_t cudaLaunchKernel(const void* func,
                             dim3 grid_dim,
                             dim3 block_dim,
                             void** args,
                             size_t shared_mem,
                             cudaStream_t stream) {
    if (func == nullptr || grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const auto* kernel = static_cast<const cumetalKernel_t*>(func);
    if (kernel->metallib_path == nullptr || kernel->kernel_name == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (kernel->arg_count > 31) {
        return fail(cudaErrorInvalidValue);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (stream != nullptr && !resolve_stream_handle(stream, &backend_stream)) {
        return fail(cudaErrorInvalidValue);
    }

    std::vector<cumetal::metal_backend::KernelArg> launch_args;
    launch_args.reserve(kernel->arg_count);

    RuntimeState& state = runtime_state();

    for (std::uint32_t i = 0; i < kernel->arg_count; ++i) {
        if (args == nullptr || args[i] == nullptr) {
            return fail(cudaErrorInvalidValue);
        }

        cumetalKernelArgInfo_t info{
            .kind = CUMETAL_ARG_BUFFER,
            .size_bytes = static_cast<std::uint32_t>(sizeof(void*)),
        };
        if (kernel->arg_info != nullptr) {
            info = kernel->arg_info[i];
        }

        if (info.kind == CUMETAL_ARG_BUFFER) {
            void* device_ptr = *reinterpret_cast<void**>(args[i]);
            cumetal::rt::AllocationTable::ResolvedAllocation resolved;
            if (!state.allocations.resolve(device_ptr, &resolved)) {
                return fail(cudaErrorInvalidDevicePointer);
            }

            cumetal::metal_backend::KernelArg arg;
            arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
            arg.buffer = std::move(resolved.buffer);
            arg.offset = resolved.offset;
            launch_args.push_back(std::move(arg));
        } else {
            if (info.size_bytes == 0 || info.size_bytes > 4096) {
                return fail(cudaErrorInvalidValue);
            }

            cumetal::metal_backend::KernelArg arg;
            arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
            arg.bytes.resize(info.size_bytes);
            std::memcpy(arg.bytes.data(), args[i], info.size_bytes);
            launch_args.push_back(std::move(arg));
        }
    }

    cumetal::metal_backend::LaunchConfig config{
        .grid = grid_dim,
        .block = block_dim,
        .shared_memory_bytes = shared_mem,
    };

    std::string error;
    const cudaError_t status =
        cumetal::metal_backend::launch_kernel(kernel->metallib_path, kernel->kernel_name, config,
                                              launch_args, backend_stream, &error);
    return fail(status);
}

cudaError_t cudaGetLastError(void) {
    const cudaError_t value = tls_last_error;
    tls_last_error = cudaSuccess;
    return value;
}

cudaError_t cudaPeekAtLastError(void) {
    return tls_last_error;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

}  // extern "C"
