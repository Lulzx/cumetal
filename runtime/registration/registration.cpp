#include "registration.h"

#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/ptx/lower_to_llvm.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::registration {

constexpr std::uint32_t kCumetalFatbinMagic = 0x4C544D43u;  // "CMTL"
constexpr std::uint32_t kCumetalFatbinVersion = 1u;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;
constexpr std::uint16_t kFatbinHeaderMinSize = 16u;
constexpr std::size_t kMaxFatbinImageBytes = 64ull * 1024ull * 1024ull;

struct CumetalFatbinImage {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    const char* metallib_path = nullptr;
};

struct FatbinWrapper {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinBlobHeader {
    std::uint32_t magic = 0;
    std::uint16_t version = 0;
    std::uint16_t header_size = 0;
    std::uint64_t fat_size = 0;
};

struct ParsedFatbinImage {
    std::string metallib_path;
    std::string ptx_source;
};

struct RegistrationModule {
    std::string metallib_path;
    std::string ptx_source;
    std::unordered_map<std::string, std::string> emitted_kernel_metallibs;
    std::vector<std::string> owned_metallibs;
};

struct RegistrationRecord {
    void* module_handle = nullptr;
    std::string metallib_path;
    std::string kernel_name;
};

struct RegistrationSymbolRecord {
    void* module_handle = nullptr;
    const void* device_address = nullptr;
    std::size_t size = 0;
};

struct RegistrationState {
    std::mutex mutex;
    std::unordered_map<void*, std::unique_ptr<RegistrationModule>> modules;
    std::unordered_map<const void*, RegistrationRecord> kernels;
    std::unordered_map<const void*, RegistrationSymbolRecord> symbols;
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

void remove_path_if_exists(const std::string& path) {
    if (path.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

bool extract_ptx_cstr(const char* chars, std::size_t max_bytes, std::string* out_ptx) {
    if (chars == nullptr || out_ptx == nullptr || max_bytes == 0) {
        return false;
    }

    const void* terminator = std::memchr(chars, '\0', max_bytes);
    if (terminator == nullptr) {
        return false;
    }

    const std::size_t size = static_cast<const char*>(terminator) - chars;
    if (size == 0) {
        return false;
    }

    const std::string candidate(chars, size);
    if (candidate.find(".version") == std::string::npos ||
        candidate.find(".entry") == std::string::npos) {
        return false;
    }

    *out_ptx = candidate;
    return true;
}

bool parse_direct_ptx_image(const void* fat_cubin, std::string* out_ptx) {
    if (fat_cubin == nullptr || out_ptx == nullptr) {
        return false;
    }

    const auto* chars = static_cast<const char*>(fat_cubin);
    if (chars[0] != '.') {
        return false;
    }

    return extract_ptx_cstr(chars, 1ull << 20, out_ptx);
}

bool extract_ptx_from_blob(const std::uint8_t* bytes,
                           std::size_t size,
                           std::string* out_ptx) {
    if (bytes == nullptr || out_ptx == nullptr || size == 0) {
        return false;
    }

    static constexpr char kMarker[] = ".version";
    constexpr std::size_t kMarkerLen = sizeof(kMarker) - 1;
    for (std::size_t i = 0; i + kMarkerLen < size; ++i) {
        if (std::memcmp(bytes + i, kMarker, kMarkerLen) != 0) {
            continue;
        }

        std::string candidate;
        if (extract_ptx_cstr(reinterpret_cast<const char*>(bytes + i), size - i, &candidate)) {
            *out_ptx = std::move(candidate);
            return true;
        }

        const char* candidate_start = reinterpret_cast<const char*>(bytes + i);
        std::size_t candidate_size = size - i;
        const void* terminator = std::memchr(candidate_start, '\0', candidate_size);
        if (terminator != nullptr) {
            candidate_size = static_cast<const char*>(terminator) - candidate_start;
        } else {
            for (std::size_t j = candidate_size; j > 0; --j) {
                if (candidate_start[j - 1] == '}') {
                    candidate_size = j;
                    break;
                }
            }
        }

        if (candidate_size == 0) {
            continue;
        }

        const std::string sliced(candidate_start, candidate_size);
        if (sliced.find(".version") == std::string::npos || sliced.find(".entry") == std::string::npos) {
            continue;
        }
        *out_ptx = sliced;
        return true;
    }
    return false;
}

bool parse_fatbin_wrapper_ptx(const void* fat_cubin, std::string* out_ptx) {
    if (fat_cubin == nullptr || out_ptx == nullptr) {
        return false;
    }

    FatbinWrapper wrapper{};
    std::memcpy(&wrapper, fat_cubin, sizeof(wrapper));
    if (wrapper.magic != kFatbinWrapperMagic || wrapper.data == nullptr) {
        return false;
    }

    if (parse_direct_ptx_image(wrapper.data, out_ptx)) {
        return true;
    }

    const auto* blob = static_cast<const std::uint8_t*>(wrapper.data);
    FatbinBlobHeader header{};
    std::memcpy(&header, blob, sizeof(header));
    if (header.magic != kFatbinBlobMagic || header.header_size < kFatbinHeaderMinSize) {
        return false;
    }

    const std::size_t header_size = static_cast<std::size_t>(header.header_size);
    const std::size_t fat_size = static_cast<std::size_t>(header.fat_size);
    if (fat_size == 0 || header_size > kMaxFatbinImageBytes || fat_size > kMaxFatbinImageBytes ||
        header_size > (kMaxFatbinImageBytes - fat_size)) {
        return false;
    }

    return extract_ptx_from_blob(blob + header_size, fat_size, out_ptx);
}

ParsedFatbinImage parse_fatbin_image(const void* fat_cubin) {
    ParsedFatbinImage parsed;
    if (fat_cubin == nullptr) {
        parsed.metallib_path = fallback_metallib_path_from_env();
        return parsed;
    }

    CumetalFatbinImage image{};
    std::memcpy(&image, fat_cubin, sizeof(image));
    if (image.magic == kCumetalFatbinMagic && image.version == kCumetalFatbinVersion &&
        image.metallib_path != nullptr) {
        parsed.metallib_path = image.metallib_path;
        return parsed;
    }

    if (parse_fatbin_wrapper_ptx(fat_cubin, &parsed.ptx_source)) {
        return parsed;
    }

    if (parse_direct_ptx_image(fat_cubin, &parsed.ptx_source)) {
        return parsed;
    }

    parsed.metallib_path = fallback_metallib_path_from_env();
    return parsed;
}

bool emit_ptx_entry_to_temp_metallib(const std::string& ptx_source,
                                     const std::string& kernel_name,
                                     std::string* out_path) {
    if (ptx_source.empty() || kernel_name.empty() || out_path == nullptr) {
        return false;
    }

    cumetal::ptx::LowerToLlvmOptions lower_options;
    lower_options.entry_name = kernel_name;
    const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx_source, lower_options);
    if (!lowered.ok || lowered.llvm_ir.empty()) {
        return false;
    }

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path ll_path =
        std::filesystem::temp_directory_path() / ("cumetal-registration-" + std::to_string(stamp) + ".ll");
    const std::filesystem::path metallib_path =
        std::filesystem::temp_directory_path() / ("cumetal-registration-" + std::to_string(stamp) + ".metallib");

    std::string io_error;
    const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
    if (!cumetal::common::write_file_bytes(ll_path, ll_bytes, &io_error)) {
        return false;
    }

    cumetal::air_emitter::EmitOptions emit_options;
    emit_options.input = ll_path;
    emit_options.output = metallib_path;
    emit_options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    emit_options.overwrite = true;
    emit_options.validate_output = false;
    emit_options.fallback_to_experimental = true;
    emit_options.kernel_name = lowered.entry_name.empty() ? kernel_name : lowered.entry_name;

    const auto emitted = cumetal::air_emitter::emit_metallib(emit_options);
    remove_path_if_exists(ll_path.string());
    if (!emitted.ok || emitted.output.empty()) {
        remove_path_if_exists(metallib_path.string());
        return false;
    }

    *out_path = emitted.output.string();
    return true;
}

std::string resolve_metallib_path_for_kernel(void* module_handle, const std::string& kernel_name) {
    if (module_handle == nullptr || kernel_name.empty()) {
        return fallback_metallib_path_from_env();
    }

    std::string ptx_source;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.modules.find(module_handle);
        if (found == s.modules.end()) {
            return fallback_metallib_path_from_env();
        }

        const RegistrationModule& module = *found->second;
        if (!module.metallib_path.empty()) {
            return module.metallib_path;
        }

        const auto cached = module.emitted_kernel_metallibs.find(kernel_name);
        if (cached != module.emitted_kernel_metallibs.end()) {
            return cached->second;
        }

        ptx_source = module.ptx_source;
    }

    if (ptx_source.empty()) {
        return fallback_metallib_path_from_env();
    }

    std::string emitted_path;
    if (!emit_ptx_entry_to_temp_metallib(ptx_source, kernel_name, &emitted_path)) {
        return fallback_metallib_path_from_env();
    }

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.modules.find(module_handle);
    if (found == s.modules.end()) {
        remove_path_if_exists(emitted_path);
        return fallback_metallib_path_from_env();
    }

    RegistrationModule& module = *found->second;
    if (!module.metallib_path.empty()) {
        remove_path_if_exists(emitted_path);
        return module.metallib_path;
    }

    const auto inserted = module.emitted_kernel_metallibs.emplace(kernel_name, emitted_path);
    if (!inserted.second) {
        remove_path_if_exists(emitted_path);
        return inserted.first->second;
    }

    module.owned_metallibs.push_back(emitted_path);
    return emitted_path;
}

std::vector<std::string> release_owned_metallibs_locked(RegistrationModule* module) {
    if (module == nullptr) {
        return {};
    }
    std::vector<std::string> owned = std::move(module->owned_metallibs);
    module->emitted_kernel_metallibs.clear();
    return owned;
}

void remove_owned_metallibs(const std::vector<std::string>& owned) {
    for (const std::string& path : owned) {
        remove_path_if_exists(path);
    }
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

bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size) {
    if (host_symbol == nullptr || out_device_symbol == nullptr) {
        return false;
    }

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.symbols.find(host_symbol);
    if (found == s.symbols.end() || found->second.device_address == nullptr) {
        return false;
    }

    *out_device_symbol = found->second.device_address;
    if (out_size != nullptr) {
        *out_size = found->second.size;
    }
    return true;
}

void clear() {
    std::vector<std::string> owned;
    RegistrationState& s = state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        for (auto& [handle, module] : s.modules) {
            (void)handle;
            if (module) {
                std::vector<std::string> module_owned =
                    release_owned_metallibs_locked(module.get());
                owned.insert(owned.end(),
                             std::make_move_iterator(module_owned.begin()),
                             std::make_move_iterator(module_owned.end()));
            }
        }
        s.kernels.clear();
        s.symbols.clear();
        s.modules.clear();
        tls_launch_stack.clear();
    }
    remove_owned_metallibs(owned);
}

}  // namespace cumetal::registration

extern "C" {

void** __cudaRegisterFatBinary(const void* fat_cubin) {
    auto module = std::make_unique<cumetal::registration::RegistrationModule>();
    const cumetal::registration::ParsedFatbinImage parsed =
        cumetal::registration::parse_fatbin_image(fat_cubin);
    module->metallib_path = parsed.metallib_path;
    module->ptx_source = parsed.ptx_source;

    if (module->metallib_path.empty() && module->ptx_source.empty()) {
        module->metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    void* handle = module.get();

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.modules.emplace(handle, std::move(module));
    return reinterpret_cast<void**>(handle);
}

void** __cudaRegisterFatBinary2(const void* fat_cubin, ...) {
    return __cudaRegisterFatBinary(fat_cubin);
}

void** __cudaRegisterFatBinary3(const void* fat_cubin, ...) {
    return __cudaRegisterFatBinary(fat_cubin);
}

void __cudaRegisterFatBinaryEnd(void** fat_cubin_handle) {
    (void)fat_cubin_handle;
}

void __cudaUnregisterFatBinary(void** fat_cubin_handle) {
    if (fat_cubin_handle == nullptr) {
        return;
    }

    void* handle = reinterpret_cast<void*>(fat_cubin_handle);
    std::vector<std::string> owned;
    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);

        for (auto it = s.kernels.begin(); it != s.kernels.end();) {
            if (it->second.module_handle == handle) {
                it = s.kernels.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = s.symbols.begin(); it != s.symbols.end();) {
            if (it->second.module_handle == handle) {
                it = s.symbols.erase(it);
            } else {
                ++it;
            }
        }

        const auto module = s.modules.find(handle);
        if (module != s.modules.end() && module->second != nullptr) {
            owned = cumetal::registration::release_owned_metallibs_locked(module->second.get());
        }
        s.modules.erase(handle);
    }
    cumetal::registration::remove_owned_metallibs(owned);
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

    std::string metallib_path =
        cumetal::registration::resolve_metallib_path_for_kernel(handle, chosen_name);
    if (metallib_path.empty()) {
        metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
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
    (void)constant;
    (void)global;

    if (host_var == nullptr) {
        return;
    }

    const void* mapped = device_address == nullptr ? static_cast<const void*>(host_var)
                                                   : static_cast<const void*>(device_address);
    void* handle = fat_cubin_handle == nullptr ? nullptr : reinterpret_cast<void*>(fat_cubin_handle);

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.symbols[host_var] = cumetal::registration::RegistrationSymbolRecord{
        .module_handle = handle,
        .device_address = mapped,
        .size = size,
    };
}

void __cudaRegisterManagedVar(void** fat_cubin_handle,
                              void** host_var_ptr_address,
                              char* device_address,
                              const char* device_name,
                              int ext,
                              std::size_t size,
                              int constant,
                              int global) {
    char* host_var = nullptr;
    if (host_var_ptr_address != nullptr) {
        host_var = static_cast<char*>(*host_var_ptr_address);
    }

    __cudaRegisterVar(fat_cubin_handle,
                      host_var,
                      device_address,
                      device_name,
                      ext,
                      size,
                      constant,
                      global);
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
