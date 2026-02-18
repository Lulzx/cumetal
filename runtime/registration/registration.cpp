#include "registration.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::registration {

constexpr std::uint32_t kCumetalFatbinMagic = 0x4C544D43u;  // "CMTL"
constexpr std::uint32_t kCumetalFatbinVersion = 1u;

struct CumetalFatbinImage {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    const char* metallib_path = nullptr;
};

struct RegistrationModule {
    std::string metallib_path;
};

struct RegistrationRecord {
    void* module_handle = nullptr;
    std::string metallib_path;
    std::string kernel_name;
};

struct RegistrationState {
    std::mutex mutex;
    std::unordered_map<void*, std::unique_ptr<RegistrationModule>> modules;
    std::unordered_map<const void*, RegistrationRecord> kernels;
};

RegistrationState& state() {
    static RegistrationState s;
    return s;
}

std::string fallback_metallib_path_from_env() {
    const char* value = std::getenv("CUMETAL_FATBIN_METALLIB");
    if (value == nullptr) {
        return {};
    }
    return std::string(value);
}

std::string extract_metallib_path(const void* fat_cubin) {
    if (fat_cubin == nullptr) {
        return fallback_metallib_path_from_env();
    }

    const auto* image = static_cast<const CumetalFatbinImage*>(fat_cubin);
    if (image->magic == kCumetalFatbinMagic && image->version == kCumetalFatbinVersion &&
        image->metallib_path != nullptr) {
        return std::string(image->metallib_path);
    }
    return fallback_metallib_path_from_env();
}

thread_local std::vector<LaunchConfiguration> tls_launch_stack;

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out) {
    if (host_function == nullptr || out == nullptr) {
        return false;
    }

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.kernels.find(host_function);
    if (found == s.kernels.end()) {
        return false;
    }

    out->metallib_path = found->second.metallib_path;
    out->kernel_name = found->second.kernel_name;
    return true;
}

void clear() {
    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.kernels.clear();
    s.modules.clear();
    tls_launch_stack.clear();
}

}  // namespace cumetal::registration

extern "C" {

void** __cudaRegisterFatBinary(const void* fat_cubin) {
    auto module = std::make_unique<cumetal::registration::RegistrationModule>();
    module->metallib_path = cumetal::registration::extract_metallib_path(fat_cubin);
    void* handle = module.get();

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.modules.emplace(handle, std::move(module));
    return reinterpret_cast<void**>(handle);
}

void __cudaUnregisterFatBinary(void** fat_cubin_handle) {
    if (fat_cubin_handle == nullptr) {
        return;
    }

    void* handle = reinterpret_cast<void*>(fat_cubin_handle);
    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);

    for (auto it = s.kernels.begin(); it != s.kernels.end();) {
        if (it->second.module_handle == handle) {
            it = s.kernels.erase(it);
        } else {
            ++it;
        }
    }
    s.modules.erase(handle);
}

void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size) {
    (void)thread_limit;
    (void)thread_id;
    (void)block_id;
    (void)block_dim;
    (void)grid_dim;
    (void)warp_size;

    if (host_function == nullptr) {
        return;
    }

    const char* chosen_name = device_name;
    if ((chosen_name == nullptr || chosen_name[0] == '\0') && device_function != nullptr &&
        device_function[0] != '\0') {
        chosen_name = device_function;
    }
    if (chosen_name == nullptr || chosen_name[0] == '\0') {
        return;
    }

    void* handle = fat_cubin_handle == nullptr ? nullptr : reinterpret_cast<void*>(fat_cubin_handle);

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);

    std::string metallib_path;
    if (handle != nullptr) {
        const auto module = s.modules.find(handle);
        if (module != s.modules.end()) {
            metallib_path = module->second->metallib_path;
        }
    }
    if (metallib_path.empty()) {
        metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    s.kernels[host_function] = cumetal::registration::RegistrationRecord{
        .module_handle = handle,
        .metallib_path = std::move(metallib_path),
        .kernel_name = chosen_name,
    };
}

void __cudaRegisterVar(void** fat_cubin_handle,
                       char* host_var,
                       char* device_address,
                       const char* device_name,
                       int ext,
                       std::size_t size,
                       int constant,
                       int global) {
    (void)fat_cubin_handle;
    (void)host_var;
    (void)device_address;
    (void)device_name;
    (void)ext;
    (void)size;
    (void)constant;
    (void)global;
}

cudaError_t __cudaPushCallConfiguration(dim3 grid_dim,
                                        dim3 block_dim,
                                        std::size_t shared_mem,
                                        cudaStream_t stream) {
    cumetal::registration::tls_launch_stack.push_back(cumetal::registration::LaunchConfiguration{
        .grid_dim = grid_dim,
        .block_dim = block_dim,
        .shared_mem = shared_mem,
        .stream = stream,
    });
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3* grid_dim,
                                       dim3* block_dim,
                                       std::size_t* shared_mem,
                                       void** stream) {
    if (cumetal::registration::tls_launch_stack.empty()) {
        return cudaErrorInvalidValue;
    }

    const cumetal::registration::LaunchConfiguration config =
        cumetal::registration::tls_launch_stack.back();
    cumetal::registration::tls_launch_stack.pop_back();

    if (grid_dim != nullptr) {
        *grid_dim = config.grid_dim;
    }
    if (block_dim != nullptr) {
        *block_dim = config.block_dim;
    }
    if (shared_mem != nullptr) {
        *shared_mem = config.shared_mem;
    }
    if (stream != nullptr) {
        *stream = reinterpret_cast<void*>(config.stream);
    }

    return cudaSuccess;
}

}  // extern "C"
