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

struct KernelMetadataField {
    std::string key;
    std::string value;
};

struct KernelRecord {
    std::string name;
    std::uint64_t bitcode_offset = 0;
    std::uint64_t bitcode_size = 0;
    bool bitcode_signature_ok = false;
    std::vector<KernelMetadataField> metadata;
};

struct CumetalContainerInfo {
    bool parsed = false;
    std::uint32_t version = 0;
    std::uint32_t flags = 0;
    std::uint32_t function_table_offset = 0;
    std::uint32_t function_count = 0;
    std::uint32_t string_table_offset = 0;
    std::uint32_t string_table_size = 0;
    std::uint32_t payload_offset = 0;
    std::uint32_t payload_size = 0;
    std::vector<KernelRecord> kernels;
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
    bool function_list_parsed = false;
    std::string function_list_parser;
    std::vector<KernelRecord> kernels;
    CumetalContainerInfo cumetal;
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
