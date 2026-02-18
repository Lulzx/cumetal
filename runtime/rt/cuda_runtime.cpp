#include "cuda_runtime.h"

#include "allocation_table.h"
#include "library_conflict.h"
#include "metal_backend.h"
#include "registration.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
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

constexpr int kCudaCompatVersion = 12000;

struct RuntimeState {
    std::once_flag init_once;
    cudaError_t init_status = cudaSuccess;
    std::string init_error;
    int current_device = 0;
    unsigned int device_flags = cudaDeviceScheduleAuto;
    cumetal::rt::AllocationTable allocations;
    std::mutex stream_mutex;
    std::unordered_map<cudaStream_t, std::shared_ptr<cumetal::metal_backend::Stream>> streams;
};

RuntimeState& runtime_state() {
    static RuntimeState state;
    return state;
}

thread_local cudaError_t tls_last_error = cudaSuccess;
thread_local std::shared_ptr<cumetal::metal_backend::Stream> tls_per_thread_stream;

void set_last_error(cudaError_t error) {
    tls_last_error = error;
}

cudaError_t ensure_initialized() {
    RuntimeState& state = runtime_state();
    std::call_once(state.init_once, [&state]() {
        const std::string conflict_warning =
            cumetal::error::detect_loaded_libcuda_conflict(reinterpret_cast<const void*>(&cudaInit));
        if (!conflict_warning.empty()) {
            std::fprintf(stderr, "%s\n", conflict_warning.c_str());
        }

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

bool is_legacy_stream_handle(cudaStream_t stream) {
    return stream == nullptr || stream == cudaStreamLegacy;
}

bool is_per_thread_stream_handle(cudaStream_t stream) {
    return stream == cudaStreamPerThread;
}

cudaError_t ensure_per_thread_stream(std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (tls_per_thread_stream != nullptr) {
        *out_stream = tls_per_thread_stream;
        return cudaSuccess;
    }

    std::string error;
    std::shared_ptr<cumetal::metal_backend::Stream> created;
    const cudaError_t status = cumetal::metal_backend::create_stream(&created, &error);
    if (status != cudaSuccess || created == nullptr) {
        return status == cudaSuccess ? cudaErrorUnknown : status;
    }
    tls_per_thread_stream = created;
    *out_stream = std::move(created);
    return cudaSuccess;
}

cudaError_t resolve_runtime_stream(cudaStream_t stream,
                                   std::shared_ptr<cumetal::metal_backend::Stream>* out_stream,
                                   bool* is_legacy_stream) {
    if (out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    out_stream->reset();
    if (is_legacy_stream != nullptr) {
        *is_legacy_stream = false;
    }

    if (is_legacy_stream_handle(stream)) {
        if (is_legacy_stream != nullptr) {
            *is_legacy_stream = true;
        }
        return cudaSuccess;
    }

    if (is_per_thread_stream_handle(stream)) {
        return ensure_per_thread_stream(out_stream);
    }

    if (!resolve_stream_handle(stream, out_stream)) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
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

cudaError_t validate_host_alloc_flags(unsigned int flags) {
    constexpr unsigned int kSupportedHostAllocFlags =
        cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;
    if ((flags & ~kSupportedHostAllocFlags) != 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t validate_device_flags(unsigned int flags) {
    constexpr unsigned int kSupportedDeviceFlags =
        cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync |
        cudaDeviceMapHost | cudaDeviceLmemResizeToMax;

    if ((flags & ~kSupportedDeviceFlags) != 0) {
        return cudaErrorInvalidValue;
    }

    const unsigned int schedule_bits =
        flags & (cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync);
    if (schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleYield) ||
        schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleBlockingSync) ||
        schedule_bits == (cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync) ||
        schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleYield |
                          cudaDeviceScheduleBlockingSync)) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

bool is_device_pointer(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    return state.allocations.resolve(ptr, &resolved);
}

cudaError_t resolve_memcpy_kind(void* dst, const void* src, cudaMemcpyKind kind, cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const cudaError_t kind_status = validate_memcpy_kind(kind);
    if (kind_status != cudaSuccess) {
        return kind_status;
    }

    const bool dst_is_device = is_device_pointer(dst);
    const bool src_is_device = is_device_pointer(src);

    if (kind == cudaMemcpyDefault) {
        if (dst_is_device && src_is_device) {
            *resolved_kind = cudaMemcpyDeviceToDevice;
        } else if (dst_is_device && !src_is_device) {
            *resolved_kind = cudaMemcpyHostToDevice;
        } else if (!dst_is_device && src_is_device) {
            *resolved_kind = cudaMemcpyDeviceToHost;
        } else {
            *resolved_kind = cudaMemcpyHostToHost;
        }
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyHostToHost:
            break;
        case cudaMemcpyHostToDevice:
            if (!dst_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDeviceToHost:
            if (!src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDeviceToDevice:
            if (!dst_is_device || !src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDefault:
            break;
    }

    *resolved_kind = kind;
    return cudaSuccess;
}

cudaError_t resolve_memcpy_to_symbol_kind(const void* src,
                                          cudaMemcpyKind kind,
                                          cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const bool src_is_device = is_device_pointer(src);
    if (kind == cudaMemcpyDefault) {
        *resolved_kind = src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyHostToDevice:
            *resolved_kind = cudaMemcpyHostToDevice;
            return cudaSuccess;
        case cudaMemcpyDeviceToDevice:
            if (!src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            *resolved_kind = cudaMemcpyDeviceToDevice;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t resolve_memcpy_from_symbol_kind(void* dst,
                                            cudaMemcpyKind kind,
                                            cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const bool dst_is_device = is_device_pointer(dst);
    if (kind == cudaMemcpyDefault) {
        *resolved_kind = dst_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyDeviceToHost:
            *resolved_kind = cudaMemcpyDeviceToHost;
            return cudaSuccess;
        case cudaMemcpyDeviceToDevice:
            if (!dst_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            *resolved_kind = cudaMemcpyDeviceToDevice;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t checked_symbol_ptr(const void* symbol,
                               size_t count,
                               size_t offset,
                               const unsigned char** out_ptr) {
    if (symbol == nullptr || out_ptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (offset > (std::numeric_limits<size_t>::max() - count)) {
        return cudaErrorInvalidValue;
    }

    *out_ptr = static_cast<const unsigned char*>(symbol) + offset;
    return cudaSuccess;
}

cudaError_t synchronize_stream_for_host_op(cudaStream_t stream,
                                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return resolve_status;
    }

    if (legacy_stream) {
        std::string error;
        return cumetal::metal_backend::synchronize(&error);
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

int cumetalRuntimeIsDevicePointer(const void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    return state.allocations.resolve(ptr, &resolved) ? 1 : 0;
}

cudaError_t cudaInit(unsigned int flags) {
    if (flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t status = ensure_initialized();
    return fail(status);
}

cudaError_t cudaDriverGetVersion(int* driver_version) {
    if (driver_version == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *driver_version = kCudaCompatVersion;
    return fail(cudaSuccess);
}

cudaError_t cudaRuntimeGetVersion(int* runtime_version) {
    if (runtime_version == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *runtime_version = kCudaCompatVersion;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceCount(int* count) {
    if (count == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *count = 1;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDevice(int* device) {
    if (device == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    *device = state.current_device;
    return fail(cudaSuccess);
}

cudaError_t cudaSetDevice(int device) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (device != 0) {
        return fail(cudaErrorInvalidValue);
    }

    RuntimeState& state = runtime_state();
    state.current_device = device;
    return fail(cudaSuccess);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const cudaError_t flags_status = validate_device_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
    }

    RuntimeState& state = runtime_state();
    state.device_flags = flags;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    if (flags == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    *flags = state.device_flags;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (prop == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (device != 0) {
        return fail(cudaErrorInvalidValue);
    }

    cumetal::metal_backend::DeviceProperties backend_props;
    std::string error;
    const cudaError_t query_status =
        cumetal::metal_backend::query_device_properties(&backend_props, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }

    std::memset(prop, 0, sizeof(*prop));
    std::strncpy(prop->name, backend_props.name.c_str(), sizeof(prop->name) - 1);
    prop->name[sizeof(prop->name) - 1] = '\0';
    prop->totalGlobalMem = backend_props.total_global_mem;
    prop->warpSize = 32;
    prop->multiProcessorCount = backend_props.multi_processor_count;
    prop->maxThreadsPerBlock =
        backend_props.max_threads_per_block > 0 ? backend_props.max_threads_per_block : 1024;
    prop->maxThreadsDim[0] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[1] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[2] = prop->maxThreadsPerBlock;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->sharedMemPerBlock =
        backend_props.shared_mem_per_block > 0 ? backend_props.shared_mem_per_block : (32 * 1024);
    prop->regsPerBlock = 65536;
    prop->major = 8;
    prop->minor = 0;

    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
    if (value == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        return fail(status);
    }

    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock:
            *value = prop.maxThreadsPerBlock;
            break;
        case cudaDevAttrMaxBlockDimX:
            *value = prop.maxThreadsDim[0];
            break;
        case cudaDevAttrMaxBlockDimY:
            *value = prop.maxThreadsDim[1];
            break;
        case cudaDevAttrMaxBlockDimZ:
            *value = prop.maxThreadsDim[2];
            break;
        case cudaDevAttrMaxGridDimX:
            *value = prop.maxGridSize[0];
            break;
        case cudaDevAttrMaxGridDimY:
            *value = prop.maxGridSize[1];
            break;
        case cudaDevAttrMaxGridDimZ:
            *value = prop.maxGridSize[2];
            break;
        case cudaDevAttrMaxSharedMemoryPerBlock:
            *value = prop.sharedMemPerBlock;
            break;
        case cudaDevAttrWarpSize:
            *value = prop.warpSize;
            break;
        case cudaDevAttrMultiProcessorCount:
            *value = prop.multiProcessorCount;
            break;
        case cudaDevAttrUnifiedAddressing:
        case cudaDevAttrManagedMemory:
        case cudaDevAttrConcurrentManagedAccess:
            *value = 1;
            break;
        default:
            return fail(cudaErrorInvalidValue);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cumetal::metal_backend::DeviceProperties backend_props;
    std::string error;
    const cudaError_t query_status =
        cumetal::metal_backend::query_device_properties(&backend_props, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }

    RuntimeState& state = runtime_state();
    const std::size_t allocated_bytes = state.allocations.total_allocated_size();
    const std::size_t total_mem = backend_props.total_global_mem;

    *total_bytes = total_mem;
    *free_bytes = allocated_bytes >= total_mem ? 0 : (total_mem - allocated_bytes);
    return fail(cudaSuccess);
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
    if (!state.allocations.insert(base, size, cumetal::rt::AllocationKind::kDevice,
                                  /*host_alloc_flags=*/0,
                                  std::move(buffer), &error)) {
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

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    if (ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t flags_status = validate_host_alloc_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
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
    if (!state.allocations.insert(base, size, cumetal::rt::AllocationKind::kHost, flags,
                                  std::move(buffer), &error)) {
        return fail(cudaErrorMemoryAllocation);
    }

    *ptr = base;
    return fail(cudaSuccess);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

cudaError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags) {
    if (dev_ptr == nullptr || host_ptr == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(host_ptr, &resolved) || resolved.offset != 0 ||
        resolved.kind != cumetal::rt::AllocationKind::kHost) {
        return fail(cudaErrorInvalidValue);
    }

    *dev_ptr = host_ptr;
    return fail(cudaSuccess);
}

cudaError_t cudaHostGetFlags(unsigned int* flags, void* host_ptr) {
    if (flags == nullptr || host_ptr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(host_ptr, &resolved) || resolved.offset != 0 ||
        resolved.kind != cumetal::rt::AllocationKind::kHost) {
        return fail(cudaErrorInvalidValue);
    }

    *flags = resolved.host_alloc_flags;
    return fail(cudaSuccess);
}

cudaError_t cudaFreeHost(void* ptr) {
    return cudaFree(ptr);
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

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
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

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyToSymbol(const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               cudaMemcpyKind kind) {
    if (symbol == nullptr || (src == nullptr && count > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_to_symbol_kind(src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(const_cast<unsigned char*>(symbol_ptr), src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyFromSymbol(void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 cudaMemcpyKind kind) {
    if ((dst == nullptr && count > 0) || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_from_symbol_kind(dst, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, symbol_ptr, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol,
                                    const void* src,
                                    size_t count,
                                    size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream) {
    if (symbol == nullptr || (src == nullptr && count > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_to_symbol_kind(src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(const_cast<unsigned char*>(symbol_ptr), src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst,
                                      const void* symbol,
                                      size_t count,
                                      size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    if ((dst == nullptr && count > 0) || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_from_symbol_kind(dst, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, symbol_ptr, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemset(void* dev_ptr, int value, size_t count) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
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

cudaError_t cudaDeviceReset(void) {
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
    std::vector<std::pair<cudaStream_t, std::shared_ptr<cumetal::metal_backend::Stream>>> streams;
    {
        std::lock_guard<std::mutex> lock(state.stream_mutex);
        streams.reserve(state.streams.size());
        for (auto& [handle, backend_stream] : state.streams) {
            streams.emplace_back(handle, std::move(backend_stream));
        }
        state.streams.clear();
    }

    for (auto& [handle, backend_stream] : streams) {
        if (backend_stream != nullptr) {
            std::string destroy_error;
            (void)cumetal::metal_backend::destroy_stream(backend_stream, &destroy_error);
        }
        delete handle;
    }

    if (tls_per_thread_stream != nullptr) {
        std::string destroy_error;
        (void)cumetal::metal_backend::destroy_stream(tls_per_thread_stream, &destroy_error);
        tls_per_thread_stream.reset();
    }

    state.allocations.clear();
    cumetal::registration::clear();
    state.current_device = 0;
    state.device_flags = cudaDeviceScheduleAuto;
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
    return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    if (stream == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (flags != cudaStreamDefault && flags != cudaStreamNonBlocking) {
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
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
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
    if (is_legacy_stream_handle(stream)) {
        return cudaDeviceSynchronize();
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
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

    if (is_legacy_stream_handle(stream)) {
        std::vector<std::shared_ptr<cumetal::metal_backend::Stream>> streams;
        {
            RuntimeState& state = runtime_state();
            std::lock_guard<std::mutex> lock(state.stream_mutex);
            streams.reserve(state.streams.size());
            for (const auto& it : state.streams) {
                streams.push_back(it.second);
            }
        }
        if (tls_per_thread_stream != nullptr) {
            streams.push_back(tls_per_thread_stream);
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
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
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

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* user_data,
                                  unsigned int flags) {
    if (callback == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    if (!legacy_stream) {
        std::string error;
        const cudaError_t ticket_status =
            cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
        if (ticket_status != cudaSuccess) {
            return fail(ticket_status);
        }
    }

    std::thread([stream, callback, user_data, backend_stream, tail_ticket, legacy_stream]() mutable {
        std::string error;
        cudaError_t callback_status = cudaSuccess;
        if (legacy_stream) {
            callback_status = cumetal::metal_backend::synchronize(&error);
        } else {
            callback_status =
                cumetal::metal_backend::stream_wait_ticket(backend_stream, tail_ticket, &error);
        }
        callback(stream, callback_status, user_data);
    }).detach();

    return fail(cudaSuccess);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    if (event == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    if (!is_legacy_stream_handle(stream)) {
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
        if (resolve_status != cudaSuccess || backend_stream == nullptr) {
            return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
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
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    bool complete = true;
    if (!legacy_stream) {
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
    } else {
        std::string error;
        const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
        if (sync_status != cudaSuccess) {
            return fail(sync_status);
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

    cumetal::registration::RegisteredKernel registered_kernel;
    const bool use_registered_kernel =
        cumetal::registration::lookup_registered_kernel(func, &registered_kernel);

    cumetalKernel_t kernel_copy{};
    const cumetalKernel_t* kernel = nullptr;
    std::uint32_t arg_count = 0;
    const cumetalKernelArgInfo_t* arg_info = nullptr;

    if (use_registered_kernel) {
        if (registered_kernel.metallib_path.empty() || registered_kernel.kernel_name.empty() ||
            args == nullptr) {
            return fail(cudaErrorInvalidValue);
        }

        std::size_t inferred_count = 0;
        for (; inferred_count < 31; ++inferred_count) {
            if (args[inferred_count] == nullptr) {
                break;
            }
        }
        if (inferred_count == 31) {
            return fail(cudaErrorInvalidValue);
        }
        arg_count = static_cast<std::uint32_t>(inferred_count);
    } else {
        std::memcpy(&kernel_copy, func, sizeof(kernel_copy));
        kernel = &kernel_copy;
        if (kernel->metallib_path == nullptr || kernel->kernel_name == nullptr || kernel->arg_count > 31) {
            return fail(cudaErrorInvalidValue);
        }
        if (kernel->arg_count > 0 && args == nullptr) {
            return fail(cudaErrorInvalidValue);
        }
        arg_count = kernel->arg_count;
        arg_info = kernel->arg_info;
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }
    if (legacy_stream) {
        std::string error;
        const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
        if (sync_status != cudaSuccess) {
            return fail(sync_status);
        }
    }

    std::vector<cumetal::metal_backend::KernelArg> launch_args;
    launch_args.reserve(arg_count);

    RuntimeState& state = runtime_state();

    for (std::uint32_t i = 0; i < arg_count; ++i) {
        if (args == nullptr || args[i] == nullptr) {
            return fail(cudaErrorInvalidValue);
        }

        cumetalKernelArgInfo_t info{
            .kind = CUMETAL_ARG_BUFFER,
            .size_bytes = static_cast<std::uint32_t>(sizeof(void*)),
        };
        if (!use_registered_kernel && arg_info != nullptr) {
            info = arg_info[i];
        } else if (use_registered_kernel) {
            std::uintptr_t value = 0;
            std::memcpy(&value, args[i], sizeof(value));
            cumetal::rt::AllocationTable::ResolvedAllocation resolved_ptr;
            if (!state.allocations.resolve(reinterpret_cast<void*>(value), &resolved_ptr)) {
                info.kind = CUMETAL_ARG_BYTES;
                info.size_bytes = value <= 0xFFFFFFFFull ? 4u : 8u;
            }
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

    const char* metallib_path =
        use_registered_kernel ? registered_kernel.metallib_path.c_str() : kernel->metallib_path;
    const char* kernel_name =
        use_registered_kernel ? registered_kernel.kernel_name.c_str() : kernel->kernel_name;

    std::string error;
    const cudaError_t status =
        cumetal::metal_backend::launch_kernel(metallib_path, kernel_name, config, launch_args,
                                              backend_stream, &error);
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

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
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
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

cudaError_t cudaProfilerStart(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaProfilerStop(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

}  // extern "C"
