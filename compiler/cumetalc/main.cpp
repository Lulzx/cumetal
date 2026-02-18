#include "cumetal/air_emitter/emitter.h"

#include <iostream>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--input] <file.{metal,ll,air,bc}> [--output|-o <file.metallib>]"
                 " [--mode xcrun|experimental] [--fallback-experimental]"
                 " [--overwrite] [--skip-validate] [--xcrun-validate]"
                 " [--kernel-name name]\n";
}

}  // namespace

int main(int argc, char** argv) {
    cumetal::air_emitter::EmitOptions options;
    bool mode_set = false;
    bool positional_input_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            if (i + 1 >= argc) {
                std::cerr << "--input expects a path\n";
                return 2;
            }
            options.input = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 >= argc) {
                std::cerr << arg << " expects a path\n";
                return 2;
            }
            options.output = argv[++i];
        } else if (arg == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "--mode expects xcrun or experimental\n";
                return 2;
            }
            const std::string mode = argv[++i];
            if (mode == "xcrun") {
                options.mode = cumetal::air_emitter::EmitMode::kXcrun;
                mode_set = true;
            } else if (mode == "experimental") {
                options.mode = cumetal::air_emitter::EmitMode::kExperimentalContainer;
                mode_set = true;
            } else {
                std::cerr << "invalid --mode: " << mode << "\n";
                return 2;
            }
        } else if (arg == "--fallback-experimental") {
            options.fallback_to_experimental = true;
        } else if (arg == "--overwrite") {
            options.overwrite = true;
        } else if (arg == "--skip-validate") {
            options.validate_output = false;
        } else if (arg == "--xcrun-validate") {
            options.run_xcrun_validate = true;
        } else if (arg == "--kernel-name") {
            if (i + 1 >= argc) {
                std::cerr << "--kernel-name expects a value\n";
                return 2;
            }
            options.kernel_name = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            return 2;
        } else if (!positional_input_set && options.input.empty()) {
            options.input = arg;
            positional_input_set = true;
        } else {
            std::cerr << "unexpected positional argument: " << arg << "\n";
            return 2;
        }
    }

    if (options.input.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    if (options.output.empty()) {
        options.output = options.input;
        options.output.replace_extension(".metallib");
    }

    if (!mode_set) {
        options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    }

    const auto result = cumetal::air_emitter::emit_metallib(options);
    for (const auto& log : result.logs) {
        if (!log.empty()) {
            std::cerr << log;
            if (log.back() != '\n') {
                std::cerr << '\n';
            }
        }
    }

    if (!result.ok) {
        std::cerr << "cumetalc failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "wrote " << result.output << "\n";
    return 0;
}
