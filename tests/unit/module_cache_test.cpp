#include "module_cache.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool read_bytes(const std::filesystem::path& path, std::vector<std::uint8_t>* out) {
    if (out == nullptr) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

}  // namespace

int main() {
    const char* old_cache = std::getenv("CUMETAL_CACHE_DIR");
    const bool had_old_cache = old_cache != nullptr;
    const std::string old_cache_value = had_old_cache ? std::string(old_cache) : std::string();

    const auto nonce = std::to_string(static_cast<long long>(
        std::filesystem::file_time_type::clock::now().time_since_epoch().count()));
    const std::filesystem::path cache_root =
        std::filesystem::temp_directory_path() / ("cumetal-module-cache-test-" + nonce);
    const std::string cache_root_str = cache_root.string();
    if (setenv("CUMETAL_CACHE_DIR", cache_root_str.c_str(), 1) != 0) {
        std::fprintf(stderr, "FAIL: setenv(CUMETAL_CACHE_DIR) failed\n");
        return 1;
    }

    std::vector<std::uint8_t> bytes = {
        'M', 'T', 'L', 'B', 0x01, 0x02, 0x03, 0x04, 0x44, 0x55, 0x66, 0x77,
    };

    std::filesystem::path staged_a;
    std::string error;
    if (!cumetal::cache::stage_metallib_bytes(bytes.data(), bytes.size(), &staged_a, &error)) {
        std::fprintf(stderr, "FAIL: stage_metallib_bytes first call failed: %s\n", error.c_str());
        return 1;
    }
    if (!std::filesystem::exists(staged_a)) {
        std::fprintf(stderr, "FAIL: staged cache path does not exist\n");
        return 1;
    }

    std::vector<std::uint8_t> roundtrip;
    if (!read_bytes(staged_a, &roundtrip) || roundtrip != bytes) {
        std::fprintf(stderr, "FAIL: staged bytes do not roundtrip\n");
        return 1;
    }

    std::filesystem::path staged_b;
    if (!cumetal::cache::stage_metallib_bytes(bytes.data(), bytes.size(), &staged_b, &error)) {
        std::fprintf(stderr, "FAIL: stage_metallib_bytes second call failed: %s\n", error.c_str());
        return 1;
    }
    if (staged_a != staged_b) {
        std::fprintf(stderr, "FAIL: identical payload should resolve to same cache path\n");
        return 1;
    }

    bytes.back() ^= 0x7f;
    std::filesystem::path staged_c;
    if (!cumetal::cache::stage_metallib_bytes(bytes.data(), bytes.size(), &staged_c, &error)) {
        std::fprintf(stderr, "FAIL: stage_metallib_bytes modified payload failed: %s\n", error.c_str());
        return 1;
    }
    if (staged_c == staged_a) {
        std::fprintf(stderr, "FAIL: modified payload should not reuse same cache path\n");
        return 1;
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_root, ec);

    if (had_old_cache) {
        (void)setenv("CUMETAL_CACHE_DIR", old_cache_value.c_str(), 1);
    } else {
        (void)unsetenv("CUMETAL_CACHE_DIR");
    }

    std::printf("PASS: module cache stages deterministic paths for metallib payloads\n");
    return 0;
}
