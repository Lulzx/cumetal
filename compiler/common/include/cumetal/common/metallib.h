#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cumetal::common {

struct BitcodeSection {
    std::string kind;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    bool reliable_size = false;
    std::uint32_t wrapper_version = 0;
    std::uint32_t wrapper_offset = 0;
    std::uint32_t wrapper_size = 0;
    std::uint32_t wrapper_cpu_type = 0;
};

struct StringCandidate {
    std::uint64_t offset = 0;
    std::string value;
};

struct MetallibSummary {
    std::string path;
    std::uint64_t file_size = 0;
    std::array<std::uint8_t, 4> magic{0, 0, 0, 0};
    std::string magic_ascii;
    std::uint32_t word1_le = 0;
    std::uint32_t word2_le = 0;
    std::vector<BitcodeSection> bitcode_sections;
    std::vector<StringCandidate> strings;
};

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path, std::string* error);
bool write_file_bytes(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& data,
                      std::string* error);

MetallibSummary inspect_metallib_bytes(const std::string& path,
                                       const std::vector<std::uint8_t>& bytes,
                                       std::size_t max_strings = 64);

bool looks_like_metallib(const MetallibSummary& summary);
std::string hex_u64(std::uint64_t value, std::size_t width = 0);

}  // namespace cumetal::common
