#include "cumetal/common/metallib.h"
#include "cumetal/ptx/lower_to_llvm.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --input <file.ptx> --output <file.ll> [--entry <name>] [--strict]"
                 " [--module-id <id>] [--target-triple <triple>] [--overwrite]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path input;
    std::filesystem::path output;
    bool overwrite = false;
    cumetal::ptx::LowerToLlvmOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            if (i + 1 >= argc) {
                std::cerr << "--input expects a path\n";
                return 2;
            }
            input = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "--output expects a path\n";
                return 2;
            }
            output = argv[++i];
        } else if (arg == "--entry") {
            if (i + 1 >= argc) {
                std::cerr << "--entry expects a function name\n";
                return 2;
            }
            options.entry_name = argv[++i];
        } else if (arg == "--module-id") {
            if (i + 1 >= argc) {
                std::cerr << "--module-id expects a value\n";
                return 2;
            }
            options.module_id = argv[++i];
        } else if (arg == "--target-triple") {
            if (i + 1 >= argc) {
                std::cerr << "--target-triple expects a value\n";
                return 2;
            }
            options.target_triple = argv[++i];
        } else if (arg == "--strict") {
            options.strict = true;
        } else if (arg == "--overwrite") {
            overwrite = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return 2;
        }
    }

    if (input.empty() || output.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    if (!std::filesystem::exists(input)) {
        std::cerr << "input does not exist: " << input << "\n";
        return 1;
    }

    if (std::filesystem::exists(output) && !overwrite) {
        std::cerr << "output exists (pass --overwrite): " << output << "\n";
        return 1;
    }

    std::string io_error;
    const std::vector<std::uint8_t> bytes = cumetal::common::read_file_bytes(input, &io_error);
    if (bytes.empty()) {
        std::cerr << "failed to read input: " << (io_error.empty() ? "empty input" : io_error) << "\n";
        return 1;
    }

    const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(
        std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()), options);
    for (const auto& warning : lowered.warnings) {
        std::cerr << "warning: " << warning << "\n";
    }
    if (!lowered.ok) {
        std::cerr << "ptx2llvm failed: " << lowered.error << "\n";
        return 1;
    }

    const std::vector<std::uint8_t> out(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
    if (!cumetal::common::write_file_bytes(output, out, &io_error)) {
        std::cerr << "failed to write output: " << io_error << "\n";
        return 1;
    }

    std::cout << "wrote " << output << " for entry " << lowered.entry_name << "\n";
    return 0;
}
