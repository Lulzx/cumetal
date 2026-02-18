#include "cumetal/common/metallib.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

namespace cumetal::common {
namespace {

constexpr std::array<std::uint8_t, 4> kRawBitcodeSig{0x42, 0x43, 0xC0, 0xDE};
constexpr std::array<std::uint8_t, 4> kWrappedBitcodeSig{0xDE, 0xC0, 0x17, 0x0B};

std::optional<std::uint32_t> read_u32_le(const std::vector<std::uint8_t>& bytes,
                                         std::size_t offset) {
    if (offset + 4 > bytes.size()) {
        return std::nullopt;
    }
    std::uint32_t value = 0;
    value |= static_cast<std::uint32_t>(bytes[offset + 0]);
    value |= static_cast<std::uint32_t>(bytes[offset + 1]) << 8;
    value |= static_cast<std::uint32_t>(bytes[offset + 2]) << 16;
    value |= static_cast<std::uint32_t>(bytes[offset + 3]) << 24;
    return value;
}

bool matches_sig_at(const std::vector<std::uint8_t>& bytes,
                    std::size_t offset,
                    const std::array<std::uint8_t, 4>& sig) {
    return offset + sig.size() <= bytes.size() &&
           std::memcmp(bytes.data() + offset, sig.data(), sig.size()) == 0;
}

bool is_visible_ascii(std::uint8_t c) {
    return c >= 32 && c <= 126;
}

bool is_valid_symbol_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.' || c == '$' ||
           c == '+' || c == '-';
}

std::string sanitize_magic(const std::array<std::uint8_t, 4>& magic) {
    std::string out;
    out.reserve(4);
    for (std::uint8_t c : magic) {
        out.push_back(is_visible_ascii(c) ? static_cast<char>(c) : '.');
    }
    return out;
}

}  // namespace

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path, std::string* error) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        if (error != nullptr) {
            *error = "failed to open file: " + path.string();
        }
        return {};
    }

    const std::ifstream::pos_type end = stream.tellg();
    if (end < 0) {
        if (error != nullptr) {
            *error = "failed to determine file size: " + path.string();
        }
        return {};
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            if (error != nullptr) {
                *error = "failed to read file: " + path.string();
            }
            return {};
        }
    }

    return bytes;
}

bool write_file_bytes(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& data,
                      std::string* error) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        if (error != nullptr) {
            *error = "failed to open output file: " + path.string();
        }
        return false;
    }

    if (!data.empty()) {
        stream.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
    }
    if (!stream) {
        if (error != nullptr) {
            *error = "failed to write output file: " + path.string();
        }
        return false;
    }
    return true;
}

MetallibSummary inspect_metallib_bytes(const std::string& path,
                                       const std::vector<std::uint8_t>& bytes,
                                       std::size_t max_strings) {
    MetallibSummary summary;
    summary.path = path;
    summary.file_size = bytes.size();

    for (std::size_t i = 0; i < 4 && i < bytes.size(); ++i) {
        summary.magic[i] = bytes[i];
    }
    summary.magic_ascii = sanitize_magic(summary.magic);

    if (const auto word = read_u32_le(bytes, 4)) {
        summary.word1_le = *word;
    }
    if (const auto word = read_u32_le(bytes, 8)) {
        summary.word2_le = *word;
    }

    std::vector<std::size_t> all_sigs;

    for (std::size_t i = 0; i + 4 <= bytes.size(); ++i) {
        if (matches_sig_at(bytes, i, kWrappedBitcodeSig)) {
            BitcodeSection section;
            section.kind = "llvm-bitcode-wrapper";
            section.offset = i;

            const auto version = read_u32_le(bytes, i + 4);
            const auto inner_offset = read_u32_le(bytes, i + 8);
            const auto inner_size = read_u32_le(bytes, i + 12);
            const auto cpu_type = read_u32_le(bytes, i + 16);

            if (version && inner_offset && inner_size && cpu_type) {
                section.wrapper_version = *version;
                section.wrapper_offset = *inner_offset;
                section.wrapper_size = *inner_size;
                section.wrapper_cpu_type = *cpu_type;

                const std::uint64_t total_size =
                    static_cast<std::uint64_t>(*inner_offset) + static_cast<std::uint64_t>(*inner_size);
                section.size = total_size;
                section.reliable_size = (section.offset + section.size <= bytes.size());
            } else {
                section.size = 4;
                section.reliable_size = false;
            }

            summary.bitcode_sections.push_back(section);
            all_sigs.push_back(i);
        } else if (matches_sig_at(bytes, i, kRawBitcodeSig)) {
            all_sigs.push_back(i);
        }
    }

    std::sort(all_sigs.begin(), all_sigs.end());
    all_sigs.erase(std::unique(all_sigs.begin(), all_sigs.end()), all_sigs.end());

    for (std::size_t i = 0; i + 4 <= bytes.size(); ++i) {
        if (!matches_sig_at(bytes, i, kRawBitcodeSig)) {
            continue;
        }

        std::uint64_t next = bytes.size();
        for (std::size_t sig : all_sigs) {
            if (sig > i) {
                next = sig;
                break;
            }
        }

        BitcodeSection section;
        section.kind = "llvm-bitcode-raw";
        section.offset = i;
        section.size = next >= i ? next - i : 0;
        section.reliable_size = next > i;
        summary.bitcode_sections.push_back(section);
    }

    std::sort(summary.bitcode_sections.begin(), summary.bitcode_sections.end(),
              [](const BitcodeSection& a, const BitcodeSection& b) {
                  if (a.offset == b.offset) {
                      return a.kind < b.kind;
                  }
                  return a.offset < b.offset;
              });

    if (max_strings == 0) {
        return summary;
    }

    for (std::size_t i = 0; i < bytes.size() && summary.strings.size() < max_strings; ++i) {
        if (i > 0 && bytes[i - 1] != 0) {
            continue;
        }
        if (!is_visible_ascii(bytes[i])) {
            continue;
        }

        std::size_t j = i;
        bool seen_alpha = false;
        std::string value;
        value.reserve(64);

        while (j < bytes.size() && bytes[j] != 0 && value.size() <= 96) {
            const char c = static_cast<char>(bytes[j]);
            if (!is_visible_ascii(bytes[j]) || !is_valid_symbol_char(c)) {
                value.clear();
                break;
            }
            seen_alpha = seen_alpha || (std::isalpha(static_cast<unsigned char>(c)) != 0);
            value.push_back(c);
            ++j;
        }

        if (value.size() < 4 || value.size() > 96 || !seen_alpha) {
            continue;
        }
        if (j >= bytes.size() || bytes[j] != 0) {
            continue;
        }

        summary.strings.push_back({.offset = i, .value = value});
        i = j;
    }

    return summary;
}

bool looks_like_metallib(const MetallibSummary& summary) {
    if (summary.file_size < 16) {
        return false;
    }
    const bool magic_looks_valid = summary.magic_ascii.rfind("MTL", 0) == 0;
    return magic_looks_valid;
}

std::string hex_u64(std::uint64_t value, std::size_t width) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    if (width > 0) {
        out << std::setw(static_cast<int>(width));
    }
    out << value;
    return out.str();
}

}  // namespace cumetal::common
