#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/ptx/lower_to_llvm.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct CommandResult {
    bool started = false;
    int exit_code = -1;
    std::string output;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--input] <file.{metal,cu,ptx,ll,air,bc}> [--output|-o <file.metallib>]"
                 " [--mode xcrun|experimental] [--fallback-experimental]"
                 " [--overwrite] [--skip-validate] [--xcrun-validate]"
                 " [--kernel-name name] [--entry name] [--ptx-strict]\n";
}

std::string lower_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

std::filesystem::path make_temp_ll_path() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto pid = static_cast<long long>(::getpid());
    return std::filesystem::temp_directory_path() /
           ("cumetalc-ptx-" + std::to_string(pid) + "-" + std::to_string(now) + ".ll");
}

std::string quote_shell(const std::string& value) {
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('\'');
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

CommandResult run_command_capture(const std::string& command) {
    CommandResult result;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return result;
    }

    result.started = true;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result.output.append(buffer);
    }

    const int status = pclose(pipe);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    }

    return result;
}

bool command_exists(const std::string& name) {
    const CommandResult result = run_command_capture("command -v " + name + " >/dev/null 2>&1; echo $?");
    if (!result.started || result.exit_code != 0 || result.output.empty()) {
        return false;
    }
    return result.output[0] == '0';
}

bool xcrun_tool_exists(const std::string& tool_name) {
    const CommandResult result =
        run_command_capture("xcrun --find " + quote_shell(tool_name) + " >/dev/null 2>&1; echo $?");
    if (!result.started || result.exit_code != 0 || result.output.empty()) {
        return false;
    }
    return result.output[0] == '0';
}

}  // namespace

int main(int argc, char** argv) {
    cumetal::air_emitter::EmitOptions options;
    bool mode_set = false;
    bool positional_input_set = false;
    std::string ptx_entry_name;
    bool ptx_strict = false;

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
        } else if (arg == "--entry") {
            if (i + 1 >= argc) {
                std::cerr << "--entry expects a value\n";
                return 2;
            }
            ptx_entry_name = argv[++i];
        } else if (arg == "--ptx-strict") {
            ptx_strict = true;
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

    std::filesystem::path temp_ll;
    const std::string input_ext = lower_ext(options.input);
    if (input_ext == ".ptx") {
        std::string io_error;
        const std::vector<std::uint8_t> ptx_bytes = cumetal::common::read_file_bytes(options.input, &io_error);
        if (ptx_bytes.empty()) {
            std::cerr << "cumetalc failed: "
                      << (io_error.empty() ? "failed to read PTX input" : io_error) << "\n";
            return 1;
        }

        cumetal::ptx::LowerToLlvmOptions lower_options;
        lower_options.strict = ptx_strict;
        lower_options.entry_name = ptx_entry_name;
        const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(
            std::string_view(reinterpret_cast<const char*>(ptx_bytes.data()), ptx_bytes.size()),
            lower_options);
        for (const auto& warning : lowered.warnings) {
            std::cerr << "ptx warning: " << warning << "\n";
        }
        if (!lowered.ok) {
            std::cerr << "cumetalc failed: PTX lowering failed: " << lowered.error << "\n";
            return 1;
        }

        temp_ll = make_temp_ll_path();
        const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
        if (!cumetal::common::write_file_bytes(temp_ll, ll_bytes, &io_error)) {
            std::cerr << "cumetalc failed: failed to write temporary LLVM IR: " << io_error << "\n";
            return 1;
        }
        options.input = temp_ll;
        options.kernel_name = lowered.entry_name;
    } else if (input_ext == ".cu") {
        if (!command_exists("xcrun")) {
            std::cerr << "cumetalc failed: xcrun is required for .cu frontend compilation\n";
            return 1;
        }
        if (!xcrun_tool_exists("clang++")) {
            std::cerr << "cumetalc failed: xcrun clang++ not available for .cu frontend compilation\n";
            return 1;
        }

        temp_ll = make_temp_ll_path();
        const std::string command =
            "xcrun clang++ -std=c++20 -S -emit-llvm -x c++ "
            "-D__global__= -D__host__= -D__device__= -D__shared__= -D__constant__= "
            "-D__managed__= " +
            quote_shell(options.input.string()) + " -o " + quote_shell(temp_ll.string()) + " 2>&1";
        const CommandResult result = run_command_capture(command);
        if (!result.started || result.exit_code != 0) {
            if (!result.output.empty()) {
                std::cerr << result.output;
                if (result.output.back() != '\n') {
                    std::cerr << '\n';
                }
            }
            std::cerr << "cumetalc failed: .cu frontend compilation failed\n";
            return 1;
        }

        options.input = temp_ll;
    }

    const auto result = cumetal::air_emitter::emit_metallib(options);
    if (!temp_ll.empty()) {
        std::error_code ec;
        std::filesystem::remove(temp_ll, ec);
    }
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
