#include "cuda.h"

#include "cuda_runtime.h"

#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>

struct CUctx_st {
    CUdevice device = 0;
};

struct CUfunc_st;

struct CUmod_st {
    std::string metallib_path;
    bool owns_metallib_path = false;
    std::vector<CUfunc_st*> functions;
};

struct CUfunc_st {
    CUmod_st* module = nullptr;
    std::string kernel_name;
};

namespace {

struct DriverState {
    std::mutex mutex;
    bool initialized = false;
    std::unordered_set<CUctx_st*> contexts;
    std::unordered_set<CUmod_st*> modules;
    std::unordered_set<CUfunc_st*> functions;
    CUctx_st* current_context = nullptr;
};

struct DriverStreamCallbackPayload {
    CUstream stream = nullptr;
    CUstreamCallback callback = nullptr;
    void* user_data = nullptr;
};

DriverState& driver_state() {
    static DriverState state;
    return state;
}

CUresult map_cuda_error(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return CUDA_SUCCESS;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
            return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorMemoryAllocation:
            return CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorInitializationError:
            return CUDA_ERROR_NOT_INITIALIZED;
        case cudaErrorNotReady:
            return CUDA_ERROR_NOT_READY;
        case cudaErrorUnknown:
            return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_ERROR_UNKNOWN;
}

bool has_current_context_locked(const DriverState& state) {
    return state.current_context != nullptr;
}

bool is_valid_function_locked(const DriverState& state, CUfunction function) {
    return function != nullptr && state.functions.find(function) != state.functions.end();
}

bool is_valid_module_locked(const DriverState& state, CUmodule module) {
    return module != nullptr && state.modules.find(module) != state.modules.end();
}

CUresult create_module_from_path(const std::string& path, bool owns_path, CUmodule* out_module) {
    if (out_module == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!std::filesystem::exists(path)) {
        return CUDA_ERROR_INVALID_IMAGE;
    }

    auto* loaded = new (std::nothrow) CUmod_st{};
    if (loaded == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    loaded->metallib_path = path;
    loaded->owns_metallib_path = owns_path;

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.modules.insert(loaded);
    }

    *out_module = loaded;
    return CUDA_SUCCESS;
}

bool parse_metallib_size(const void* image, std::size_t* out_size) {
    if (image == nullptr || out_size == nullptr) {
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (std::memcmp(bytes, "MTLB", 4) != 0) {
        return false;
    }

    std::uint64_t raw_size = 0;
    std::memcpy(&raw_size, bytes + 0x10, sizeof(raw_size));
    if (raw_size < 0x20 || raw_size > (1ull << 31)) {
        return false;
    }

    *out_size = static_cast<std::size_t>(raw_size);
    return true;
}

bool stage_module_image_to_tempfile(const void* image, std::size_t size, std::string* out_path) {
    if (image == nullptr || size == 0 || out_path == nullptr) {
        return false;
    }

    const std::string filename =
        "cumetal_module_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".metallib";
    const std::filesystem::path path = std::filesystem::temp_directory_path() / filename;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out.write(static_cast<const char*>(image), static_cast<std::streamsize>(size));
    out.close();
    if (!out.good()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return false;
    }

    *out_path = path.string();
    return true;
}

void driver_stream_callback_bridge(cudaStream_t stream, cudaError_t status, void* user_data) {
    (void)stream;
    auto* payload = static_cast<DriverStreamCallbackPayload*>(user_data);
    if (payload == nullptr || payload->callback == nullptr) {
        return;
    }

    payload->callback(payload->stream, map_cuda_error(status), payload->user_data);
    delete payload;
}

}  // namespace

extern "C" {

CUresult cuInit(unsigned int flags) {
    if (flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const CUresult status = map_cuda_error(cudaInit(0));
    if (status != CUDA_SUCCESS) {
        return status;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.initialized = true;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
    if (count == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    *count = 1;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (ordinal != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (name == nullptr || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    std::strncpy(name, prop.name, static_cast<std::size_t>(len - 1));
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    if (bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *bytes = prop.totalGlobalMem;
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    auto* context = new (std::nothrow) CUctx_st{};
    if (context == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    context->device = dev;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.contexts.insert(context);
        if (state.current_context == nullptr) {
            state.current_context = context;
        }
    }

    *pctx = context;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    if (ctx == nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.contexts.find(ctx) == state.contexts.end()) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        state.contexts.erase(ctx);
        if (state.current_context == ctx) {
            state.current_context = state.contexts.empty() ? nullptr : *state.contexts.begin();
        }
    }

    delete ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (ctx != nullptr && state.contexts.find(ctx) == state.contexts.end()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    state.current_context = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    *pctx = state.current_context;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    return map_cuda_error(cudaDeviceSynchronize());
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
    if (flags != CU_STREAM_DEFAULT && flags != CU_STREAM_NON_BLOCKING) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return map_cuda_error(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(phStream), flags));
}

CUresult cuStreamDestroy(CUstream hStream) {
    return map_cuda_error(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamSynchronize(CUstream hStream) {
    return map_cuda_error(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamQuery(CUstream hStream) {
    return map_cuda_error(cudaStreamQuery(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamAddCallback(CUstream hStream,
                             CUstreamCallback callback,
                             void* userData,
                             unsigned int flags) {
    if (callback == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    auto* payload = new (std::nothrow) DriverStreamCallbackPayload{};
    if (payload == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    payload->stream = hStream;
    payload->callback = callback;
    payload->user_data = userData;

    const cudaError_t status = cudaStreamAddCallback(reinterpret_cast<cudaStream_t>(hStream),
                                                     driver_stream_callback_bridge,
                                                     payload,
                                                     flags);
    if (status != cudaSuccess) {
        delete payload;
        return map_cuda_error(status);
    }

    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
    return map_cuda_error(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(hStream),
                                              reinterpret_cast<cudaEvent_t>(hEvent),
                                              flags));
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) {
    return map_cuda_error(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(phEvent), flags));
}

CUresult cuEventDestroy(CUevent hEvent) {
    return map_cuda_error(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return map_cuda_error(cudaEventRecord(reinterpret_cast<cudaEvent_t>(hEvent),
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuEventSynchronize(CUevent hEvent) {
    return map_cuda_error(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventQuery(CUevent hEvent) {
    return map_cuda_error(cudaEventQuery(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    return map_cuda_error(cudaEventElapsedTime(pMilliseconds,
                                               reinterpret_cast<cudaEvent_t>(hStart),
                                               reinterpret_cast<cudaEvent_t>(hEnd)));
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    if (module == nullptr || fname == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    return create_module_from_path(fname, /*owns_path=*/false, module);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    if (module == nullptr || image == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    std::size_t size = 0;
    if (!parse_metallib_size(image, &size)) {
        return CUDA_ERROR_INVALID_IMAGE;
    }

    std::string staged_path;
    if (!stage_module_image_to_tempfile(image, size, &staged_path)) {
        return CUDA_ERROR_UNKNOWN;
    }

    const CUresult load_status = create_module_from_path(staged_path, /*owns_path=*/true, module);
    if (load_status != CUDA_SUCCESS) {
        std::error_code ec;
        std::filesystem::remove(staged_path, ec);
    }
    return load_status;
}

CUresult cuModuleLoadDataEx(CUmodule* module,
                            const void* image,
                            unsigned int numOptions,
                            void* options,
                            void* optionValues) {
    if (numOptions != 0 || options != nullptr || optionValues != nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    if (module == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::string owned_path;
    bool remove_owned_path = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!is_valid_module_locked(state, module)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        for (CUfunc_st* function : module->functions) {
            state.functions.erase(function);
            delete function;
        }
        module->functions.clear();
        if (module->owns_metallib_path) {
            owned_path = module->metallib_path;
            remove_owned_path = true;
        }
        state.modules.erase(module);
    }

    delete module;
    if (remove_owned_path) {
        std::error_code ec;
        std::filesystem::remove(owned_path, ec);
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (hfunc == nullptr || hmod == nullptr || name == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!is_valid_module_locked(state, hmod)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    auto* function = new (std::nothrow) CUfunc_st{};
    if (function == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    function->module = hmod;
    function->kernel_name = name;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        hmod->functions.push_back(function);
        state.functions.insert(function);
    }

    *hfunc = function;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra) {
    if (f == nullptr || gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 || blockDimX == 0 ||
        blockDimY == 0 || blockDimZ == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams != nullptr && extra != nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!is_valid_function_locked(state, f)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    std::vector<void*> launch_params;
    std::vector<CUdeviceptr> packed_arg_values;

    if (extra != nullptr) {
        void* packed_buffer = nullptr;
        std::size_t packed_size = 0;
        bool have_buffer = false;
        bool have_size = false;

        for (std::size_t i = 0;;) {
            void* key = extra[i++];
            if (key == CU_LAUNCH_PARAM_END) {
                break;
            }
            if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                if (extra[i] == nullptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                packed_buffer = extra[i++];
                have_buffer = true;
                continue;
            }
            if (key == CU_LAUNCH_PARAM_BUFFER_SIZE) {
                if (extra[i] == nullptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                packed_size = *reinterpret_cast<const std::size_t*>(extra[i]);
                have_size = true;
                ++i;
                continue;
            }
            return CUDA_ERROR_INVALID_VALUE;
        }

        if (!have_buffer || !have_size || (packed_size % sizeof(CUdeviceptr) != 0)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const std::size_t arg_count = packed_size / sizeof(CUdeviceptr);
        if (arg_count > 31) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const auto* packed_args = static_cast<const CUdeviceptr*>(packed_buffer);
        packed_arg_values.assign(packed_args, packed_args + arg_count);
        launch_params.reserve(arg_count);
        for (std::size_t i = 0; i < packed_arg_values.size(); ++i) {
            launch_params.push_back(&packed_arg_values[i]);
        }
    } else if (kernelParams != nullptr) {
        std::size_t arg_count = 0;
        for (; arg_count < 31; ++arg_count) {
            if (kernelParams[arg_count] == nullptr) {
                break;
            }
        }
        if (arg_count == 31 && kernelParams[31] != nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        launch_params.reserve(arg_count);
        for (std::size_t i = 0; i < arg_count; ++i) {
            launch_params.push_back(kernelParams[i]);
        }
    }

    std::vector<cumetalKernelArgInfo_t> arg_info;
    arg_info.reserve(launch_params.size());
    for (std::size_t i = 0; i < launch_params.size(); ++i) {
        (void)i;
        arg_info.push_back(cumetalKernelArgInfo_t{
            .kind = CUMETAL_ARG_BUFFER,
            .size_bytes = 0,
        });
    }

    const cumetalKernel_t kernel{
        .metallib_path = f->module->metallib_path.c_str(),
        .kernel_name = f->kernel_name.c_str(),
        .arg_count = static_cast<std::uint32_t>(arg_info.size()),
        .arg_info = arg_info.empty() ? nullptr : arg_info.data(),
    };

    const cudaError_t status = cudaLaunchKernel(&kernel,
                                                dim3(gridDimX, gridDimY, gridDimZ),
                                                dim3(blockDimX, blockDimY, blockDimZ),
                                                launch_params.empty() ? nullptr : launch_params.data(),
                                                sharedMemBytes,
                                                reinterpret_cast<cudaStream_t>(hStream));
    return map_cuda_error(status);
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    void* allocated = nullptr;
    const cudaError_t status = cudaMalloc(&allocated, bytesize);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocated));
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    return map_cuda_error(cudaFree(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dptr))));
}

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
    return map_cuda_error(cudaHostAlloc(pp, bytesize, cudaHostAllocDefault));
}

CUresult cuMemFreeHost(void* p) {
    return map_cuda_error(cudaFreeHost(p));
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    return map_cuda_error(cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     srcHost,
                                     ByteCount,
                                     cudaMemcpyHostToDevice));
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return map_cuda_error(cudaMemcpy(dstHost,
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                     ByteCount,
                                     cudaMemcpyDeviceToHost));
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    return map_cuda_error(cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                     ByteCount,
                                     cudaMemcpyDeviceToDevice));
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                           const void* srcHost,
                           size_t ByteCount,
                           CUstream hStream) {
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          srcHost,
                                          ByteCount,
                                          cudaMemcpyHostToDevice,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyDtoHAsync(void* dstHost,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream) {
    return map_cuda_error(cudaMemcpyAsync(dstHost,
                                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                          ByteCount,
                                          cudaMemcpyDeviceToHost,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream) {
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                          ByteCount,
                                          cudaMemcpyDeviceToDevice,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    return map_cuda_error(cudaMemset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     static_cast<int>(uc),
                                     N));
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    return map_cuda_error(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          static_cast<int>(uc),
                                          N,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
    if (pStr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    switch (error) {
        case CUDA_SUCCESS:
            *pStr = "CUDA_SUCCESS";
            break;
        case CUDA_ERROR_INVALID_VALUE:
            *pStr = "CUDA_ERROR_INVALID_VALUE";
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            *pStr = "CUDA_ERROR_NOT_INITIALIZED";
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            *pStr = "CUDA_ERROR_INVALID_DEVICE";
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            *pStr = "CUDA_ERROR_INVALID_IMAGE";
            break;
        case CUDA_ERROR_INVALID_CONTEXT:
            *pStr = "CUDA_ERROR_INVALID_CONTEXT";
            break;
        case CUDA_ERROR_NOT_FOUND:
            *pStr = "CUDA_ERROR_NOT_FOUND";
            break;
        case CUDA_ERROR_NOT_READY:
            *pStr = "CUDA_ERROR_NOT_READY";
            break;
        case CUDA_ERROR_UNKNOWN:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
        default:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return cuGetErrorName(error, pStr);
}

}  // extern "C"
