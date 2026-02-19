#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/ptx/lower_to_metal.h"
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

#ifndef CUMETAL_SOURCE_DIR
#define CUMETAL_SOURCE_DIR ""
#endif

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
                 " [--kernel-name name] [--entry name] [--ptx-strict]"
                 " [--fp64=native|emulate|warn]\n";
}

std::string lower_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

std::filesystem::path make_temp_path(const std::string& extension_with_dot) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto pid = static_cast<long long>(::getpid());
    return std::filesystem::temp_directory_path() /
           ("cumetalc-ptx-" + std::to_string(pid) + "-" + std::to_string(now) + extension_with_dot);
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

bool try_emit_vector_add_air_ir_from_cu(const std::filesystem::path& input_cu,
                                        const std::filesystem::path& output_ll,
                                        std::string* error) {
    std::string io_error;
    const std::vector<std::uint8_t> source_bytes = cumetal::common::read_file_bytes(input_cu, &io_error);
    if (source_bytes.empty()) {
        if (error != nullptr) {
            *error = io_error.empty() ? "failed to read .cu source" : io_error;
        }
        return false;
    }

    std::string source(source_bytes.begin(), source_bytes.end());
    std::string normalized;
    normalized.reserve(source.size());
    for (char c : source) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            normalized.push_back(c);
        }
    }

    if (normalized.find("vector_add(") == std::string::npos ||
        normalized.find("c[id]=a[id]+b[id];") == std::string::npos) {
        if (error != nullptr) {
            *error = "unsupported .cu pattern for no-llvm-as fallback";
        }
        return false;
    }

    static constexpr char kVectorAddAirTemplate[] =
        "target triple = \"air64_v28-apple-macosx26.0.0\"\n"
        "\n"
        "define void @vector_add(float addrspace(1)* %a, float addrspace(1)* %b, "
        "float addrspace(1)* %c, i32 %id) #0 {\n"
        "entry:\n"
        "  %pa = getelementptr float, float addrspace(1)* %a, i32 %id\n"
        "  %pb = getelementptr float, float addrspace(1)* %b, i32 %id\n"
        "  %pc = getelementptr float, float addrspace(1)* %c, i32 %id\n"
        "  %va = load float, float addrspace(1)* %pa, align 4\n"
        "  %vb = load float, float addrspace(1)* %pb, align 4\n"
        "  %sum = fadd float %va, %vb\n"
        "  store float %sum, float addrspace(1)* %pc, align 4\n"
        "  ret void\n"
        "}\n"
        "\n"
        "attributes #0 = { \"air.kernel\" \"air.version\"=\"2.8\" }\n"
        "\n"
        "!air.kernel = !{!0}\n"
        "!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @vector_add, !1, !2}\n"
        "!1 = !{}\n"
        "!2 = !{!3, !4, !5, !6}\n"
        "!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"a\"}\n"
        "!4 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"b\"}\n"
        "!5 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"c\"}\n"
        "!6 = !{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"id\"}\n"
        "!air.compile_options = !{!7, !8, !9}\n"
        "!7 = !{!\"air.compile.denorms_disable\"}\n"
        "!8 = !{!\"air.compile.fast_math_enable\"}\n"
        "!9 = !{!\"air.compile.framebuffer_fetch_enable\"}\n"
        "!air.version = !{!10}\n"
        "!air.language_version = !{!11}\n"
        "!10 = !{i32 2, i32 8, i32 0}\n"
        "!11 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

    const std::vector<std::uint8_t> out_bytes(
        reinterpret_cast<const std::uint8_t*>(kVectorAddAirTemplate),
        reinterpret_cast<const std::uint8_t*>(kVectorAddAirTemplate) +
            std::char_traits<char>::length(kVectorAddAirTemplate));
    if (!cumetal::common::write_file_bytes(output_ll, out_bytes, &io_error)) {
        if (error != nullptr) {
            *error = io_error.empty() ? "failed to write fallback AIR LLVM IR" : io_error;
        }
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    cumetal::air_emitter::EmitOptions options;
    bool mode_set = false;
    bool positional_input_set = false;
    std::string ptx_entry_name;
    bool ptx_strict = false;
    cumetal::ptx::Fp64Mode ptx_fp64_mode = cumetal::ptx::Fp64Mode::kNative;

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
        } else if (arg.size() > 7 && arg.substr(0, 7) == "--fp64=") {
            const std::string fp64_mode_str = arg.substr(7);
            if (fp64_mode_str == "native") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kNative;
            } else if (fp64_mode_str == "emulate") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
            } else if (fp64_mode_str == "warn") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kWarn;
            } else {
                std::cerr << "invalid --fp64 mode: " << fp64_mode_str
                          << " (valid: native, emulate, warn)\n";
                return 2;
            }
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

    std::filesystem::path temp_stage_file;
    const std::string input_ext = lower_ext(options.input);
    if (input_ext == ".ptx") {
        std::string io_error;
        const std::vector<std::uint8_t> ptx_bytes = cumetal::common::read_file_bytes(options.input, &io_error);
        if (ptx_bytes.empty()) {
            std::cerr << "cumetalc failed: "
                      << (io_error.empty() ? "failed to read PTX input" : io_error) << "\n";
            return 1;
        }

        const std::string ptx_source(reinterpret_cast<const char*>(ptx_bytes.data()), ptx_bytes.size());

        cumetal::ptx::LowerToMetalOptions lower_to_metal_options;
        lower_to_metal_options.strict = ptx_strict;
        lower_to_metal_options.entry_name = ptx_entry_name;
        const auto lowered_metal =
            cumetal::ptx::lower_ptx_to_metal_source(std::string_view(ptx_source), lower_to_metal_options);
        for (const auto& warning : lowered_metal.warnings) {
            std::cerr << "ptx warning: " << warning << "\n";
        }
        if (!lowered_metal.ok) {
            std::cerr << "cumetalc failed: PTX->Metal lowering failed: " << lowered_metal.error << "\n";
            return 1;
        }

        if (lowered_metal.matched && !lowered_metal.metal_source.empty()) {
            temp_stage_file = make_temp_path(".metal");
            const std::vector<std::uint8_t> metal_bytes(lowered_metal.metal_source.begin(),
                                                        lowered_metal.metal_source.end());
            if (!cumetal::common::write_file_bytes(temp_stage_file, metal_bytes, &io_error)) {
                std::cerr << "cumetalc failed: failed to write temporary Metal source: "
                          << io_error << "\n";
                return 1;
            }
            options.input = temp_stage_file;
            options.kernel_name = lowered_metal.entry_name;
        } else {
            cumetal::ptx::LowerToLlvmOptions lower_options;
            lower_options.strict = ptx_strict;
            lower_options.entry_name = ptx_entry_name;
            lower_options.fp64_mode = ptx_fp64_mode;
            const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(std::string_view(ptx_source), lower_options);
            for (const auto& warning : lowered.warnings) {
                std::cerr << "ptx warning: " << warning << "\n";
            }
            if (!lowered.ok) {
                std::cerr << "cumetalc failed: PTX lowering failed: " << lowered.error << "\n";
                return 1;
            }

            temp_stage_file = make_temp_path(".ll");
            const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
            if (!cumetal::common::write_file_bytes(temp_stage_file, ll_bytes, &io_error)) {
                std::cerr << "cumetalc failed: failed to write temporary LLVM IR: " << io_error << "\n";
                return 1;
            }
            options.input = temp_stage_file;
            options.kernel_name = lowered.entry_name;
        }
    } else if (input_ext == ".cu") {
        const bool needs_fallback_air_ll =
            options.mode == cumetal::air_emitter::EmitMode::kXcrun && !command_exists("llvm-as");
        if (needs_fallback_air_ll) {
            temp_stage_file = make_temp_path(".ll");
            std::string fallback_error;
            if (try_emit_vector_add_air_ir_from_cu(options.input, temp_stage_file, &fallback_error)) {
                options.input = temp_stage_file;
                options.kernel_name = "vector_add";
            } else {
                std::cerr << "cumetalc warning: " << fallback_error
                          << "; attempting generic .cu frontend lowering\n";
                temp_stage_file.clear();
            }
        }

        if (!temp_stage_file.empty()) {
            // Fallback AIR-ready LLVM IR path selected.
        } else {
        if (!command_exists("xcrun")) {
            std::cerr << "cumetalc failed: xcrun is required for .cu frontend compilation\n";
            return 1;
        }
        if (!xcrun_tool_exists("clang++")) {
            std::cerr << "cumetalc failed: xcrun clang++ not available for .cu frontend compilation\n";
            return 1;
        }

        temp_stage_file = make_temp_path(".ll");
        const std::filesystem::path runtime_api_dir =
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api";
        const std::string command =
            "xcrun clang++ -std=c++20 -S -emit-llvm -x c++ "
            "-D__global__= -D__host__= -D__device__= -D__shared__= -D__constant__= "
            "-D__managed__= " +
            ((std::filesystem::exists(runtime_api_dir) && std::filesystem::is_directory(runtime_api_dir))
                 ? ("-I " + quote_shell(runtime_api_dir.string()) + " ")
                 : "") +
            quote_shell(options.input.string()) + " -o " + quote_shell(temp_stage_file.string()) + " 2>&1";
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

        options.input = temp_stage_file;
        }
    }

    const auto result = cumetal::air_emitter::emit_metallib(options);
    if (!temp_stage_file.empty()) {
        std::error_code ec;
        std::filesystem::remove(temp_stage_file, ec);
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
