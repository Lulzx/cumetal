#include "cumetal/common/metallib.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <file.metallib> [--json] [--max-strings N]\n";
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    bool json = false;
    std::size_t max_strings = 64;
    std::string input;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") {
            json = true;
        } else if (arg == "--max-strings") {
            if (i + 1 >= argc) {
                std::cerr << "--max-strings expects an integer\n";
                return 2;
            }
            max_strings = std::strtoull(argv[++i], nullptr, 10);
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 2;
        } else if (input.empty()) {
            input = arg;
        } else {
            std::cerr << "Unexpected positional argument: " << arg << "\n";
            return 2;
        }
    }

    if (input.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    std::string error;
    const std::vector<std::uint8_t> bytes =
        cumetal::common::read_file_bytes(std::filesystem::path(input), &error);
    if (bytes.empty()) {
        std::cerr << "air_inspect: " << error << "\n";
        return 1;
    }

    const auto summary = cumetal::common::inspect_metallib_bytes(input, bytes, max_strings);

    if (json) {
        std::cout << "{\n";
        std::cout << "  \"file\": \"" << json_escape(summary.path) << "\",\n";
        std::cout << "  \"size\": " << summary.file_size << ",\n";
        std::cout << "  \"magic\": \"" << json_escape(summary.magic_ascii) << "\",\n";
        std::cout << "  \"word1_le\": " << summary.word1_le << ",\n";
        std::cout << "  \"word2_le\": " << summary.word2_le << ",\n";

        std::cout << "  \"bitcode_sections\": [\n";
        for (std::size_t i = 0; i < summary.bitcode_sections.size(); ++i) {
            const auto& section = summary.bitcode_sections[i];
            std::cout << "    {\"kind\": \"" << section.kind << "\", "
                      << "\"offset\": " << section.offset << ", "
                      << "\"size\": " << section.size << ", "
                      << "\"reliable_size\": " << (section.reliable_size ? "true" : "false")
                      << "}";
            if (i + 1 != summary.bitcode_sections.size()) {
                std::cout << ',';
            }
            std::cout << "\n";
        }
        std::cout << "  ],\n";

        std::cout << "  \"string_candidates\": [\n";
        for (std::size_t i = 0; i < summary.strings.size(); ++i) {
            const auto& entry = summary.strings[i];
            std::cout << "    {\"offset\": " << entry.offset << ", \"value\": \""
                      << json_escape(entry.value) << "\"}";
            if (i + 1 != summary.strings.size()) {
                std::cout << ',';
            }
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        return 0;
    }

    std::cout << "File: " << summary.path << "\n";
    std::cout << "Size: " << summary.file_size << " bytes\n";
    std::cout << "Magic: " << summary.magic_ascii << " ["
              << static_cast<int>(summary.magic[0]) << ',' << static_cast<int>(summary.magic[1])
              << ',' << static_cast<int>(summary.magic[2]) << ','
              << static_cast<int>(summary.magic[3]) << "]\n";
    std::cout << "Header words (LE): " << summary.word1_le << ", " << summary.word2_le << "\n";

    if (!cumetal::common::looks_like_metallib(summary)) {
        std::cout << "Container check: does not look like a standard metallib (magic does not start with MTL).\n";
    } else {
        std::cout << "Container check: looks like a metallib container.\n";
    }

    std::cout << "\nEmbedded bitcode sections: " << summary.bitcode_sections.size() << "\n";
    for (std::size_t i = 0; i < summary.bitcode_sections.size(); ++i) {
        const auto& section = summary.bitcode_sections[i];
        std::cout << "  [" << i << "] kind=" << section.kind
                  << " offset=" << cumetal::common::hex_u64(section.offset)
                  << " size=" << cumetal::common::hex_u64(section.size)
                  << " reliable=" << (section.reliable_size ? "yes" : "no") << "\n";
        if (section.kind == "llvm-bitcode-wrapper") {
            std::cout << "      wrapper.version=" << section.wrapper_version
                      << " wrapper.offset=" << section.wrapper_offset
                      << " wrapper.size=" << section.wrapper_size
                      << " wrapper.cpu_type=" << section.wrapper_cpu_type << "\n";
        }
    }

    std::cout << "\nString candidates: " << summary.strings.size() << "\n";
    for (const auto& entry : summary.strings) {
        std::cout << "  @" << cumetal::common::hex_u64(entry.offset) << " " << entry.value << "\n";
    }

    return 0;
}
