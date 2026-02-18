#include "cumetal/air_validate/validator.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <file.metallib> [--xcrun] [--no-bitcode-required] [--relaxed-magic]\n";
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

const char* severity_name(cumetal::air_validate::Severity severity) {
    switch (severity) {
        case cumetal::air_validate::Severity::kInfo:
            return "info";
        case cumetal::air_validate::Severity::kWarning:
            return "warning";
        case cumetal::air_validate::Severity::kError:
            return "error";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    cumetal::air_validate::ValidationOptions options;
    bool json = false;
    std::string input;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--xcrun") {
            options.run_xcrun_validate = true;
        } else if (arg == "--no-bitcode-required") {
            options.require_bitcode = false;
        } else if (arg == "--relaxed-magic") {
            options.strict_magic = false;
        } else if (arg == "--json") {
            json = true;
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

    const auto result = cumetal::air_validate::validate_file(input, options);

    if (json) {
        std::cout << "{\n";
        std::cout << "  \"ok\": " << (result.ok ? "true" : "false") << ",\n";
        std::cout << "  \"file\": \"" << json_escape(result.summary.path) << "\",\n";
        std::cout << "  \"size\": " << result.summary.file_size << ",\n";
        std::cout << "  \"magic\": \"" << json_escape(result.summary.magic_ascii) << "\",\n";
        std::cout << "  \"bitcode_sections\": " << result.summary.bitcode_sections.size() << ",\n";
        std::cout << "  \"diagnostics\": [\n";
        for (std::size_t i = 0; i < result.diagnostics.size(); ++i) {
            const auto& diagnostic = result.diagnostics[i];
            std::cout << "    {\"severity\": \"" << severity_name(diagnostic.severity)
                      << "\", \"message\": \"" << json_escape(diagnostic.message) << "\"}";
            if (i + 1 != result.diagnostics.size()) {
                std::cout << ',';
            }
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
    } else {
        std::cout << cumetal::air_validate::format_report(result);
    }

    return result.ok ? 0 : 1;
}
