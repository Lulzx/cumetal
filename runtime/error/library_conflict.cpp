#include "library_conflict.h"

#include <dlfcn.h>
#include <mach-o/dyld.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cumetal::error {
namespace {

std::string normalize_path(const std::string& path) {
    if (path.empty()) {
        return std::string();
    }

    std::error_code ec;
    const std::filesystem::path raw(path);
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(raw, ec);
    if (!ec && !canonical.empty()) {
        return canonical.string();
    }

    return raw.lexically_normal().string();
}

std::string basename_of(const std::string& path) {
    if (path.empty()) {
        return std::string();
    }
    return std::filesystem::path(path).filename().string();
}

std::vector<std::string> collect_loaded_image_paths() {
    std::vector<std::string> paths;
    const std::uint32_t image_count = _dyld_image_count();
    paths.reserve(image_count);
    for (std::uint32_t i = 0; i < image_count; ++i) {
        const char* image_name = _dyld_get_image_name(i);
        if (image_name == nullptr || image_name[0] == '\0') {
            continue;
        }
        paths.emplace_back(image_name);
    }
    return paths;
}

std::string self_image_path_for_symbol(const void* symbol) {
    if (symbol == nullptr) {
        return std::string();
    }

    Dl_info info{};
    if (dladdr(symbol, &info) == 0) {
        return std::string();
    }
    if (info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
        return std::string();
    }
    return std::string(info.dli_fname);
}

}  // namespace

std::string detect_libcuda_conflict_from_paths(const std::vector<std::string>& image_paths,
                                               const std::string& self_image_path) {
    const std::string normalized_self = normalize_path(self_image_path);
    for (const std::string& image_path : image_paths) {
        if (basename_of(image_path) != "libcuda.dylib") {
            continue;
        }

        const std::string normalized_candidate = normalize_path(image_path);
        if (!normalized_self.empty() && normalized_candidate == normalized_self) {
            continue;
        }

        std::string warning =
            "CuMetal warning: detected another libcuda.dylib loaded in this process.\n";
        warning += "  CuMetal image: " +
                   (normalized_self.empty() ? std::string("<unknown>") : normalized_self) + "\n";
        warning += "  Other libcuda: " +
                   (normalized_candidate.empty() ? image_path : normalized_candidate) + "\n";
        warning += "  This may conflict with CUDA symbol resolution.";
        return warning;
    }

    return std::string();
}

std::string detect_loaded_libcuda_conflict(const void* self_symbol) {
    const std::string self_image = self_image_path_for_symbol(self_symbol);
    const std::vector<std::string> image_paths = collect_loaded_image_paths();
    return detect_libcuda_conflict_from_paths(image_paths, self_image);
}

}  // namespace cumetal::error
