#include "cumetal/air_emitter/emitter.h"

#include "cumetal/air_validate/validator.h"
#include "cumetal/common/metallib.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace cumetal::air_emitter {
namespace {

struct CommandResult {
    bool started = false;
    int exit_code = -1;
    std::string output;
};

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

std::filesystem::path make_temp_dir() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto pid = static_cast<long long>(::getpid());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                      ("cumetal-air-emitter-" + std::to_string(pid) + "-" +
                                       std::to_string(now));
    std::filesystem::create_directories(dir);
    return dir;
}

std::string lower_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
}

EmitResult emit_experimental(const EmitOptions& options) {
    EmitResult result;
    result.mode_used = EmitMode::kExperimentalContainer;

    std::string io_error;
    const std::vector<std::uint8_t> payload =
        cumetal::common::read_file_bytes(options.input, &io_error);
    if (payload.empty()) {
        result.error = io_error.empty() ? "input payload is empty" : io_error;
        return result;
    }

    std::vector<std::uint8_t> container;
    container.reserve(64 + options.kernel_name.size() + payload.size());

    // Provisional developer-only container format used when xcrun tools are unavailable.
    // Header (little-endian):
    //   magic[4] = "MTLB"
    //   version  = 1
    //   total_size
    //   name_offset
    //   payload_offset
    //   payload_size
    //   flags
    container.push_back('M');
    container.push_back('T');
    container.push_back('L');
    container.push_back('B');
    append_u32_le(container, 1);
    append_u32_le(container, 0);  // total size (patched below)

    constexpr std::uint32_t header_size = 4 + 4 + 4 + 4 + 4 + 4 + 4;
    const std::uint32_t name_offset = header_size;
    const std::uint32_t payload_offset =
        name_offset + static_cast<std::uint32_t>(options.kernel_name.size()) + 1;

    append_u32_le(container, name_offset);
    append_u32_le(container, payload_offset);
    append_u32_le(container, static_cast<std::uint32_t>(payload.size()));
    append_u32_le(container, 0x434d544c);  // "CMTL"

    container.insert(container.end(), options.kernel_name.begin(), options.kernel_name.end());
    container.push_back(0);

    if (container.size() < payload_offset) {
        container.resize(payload_offset, 0);
    }
    container.insert(container.end(), payload.begin(), payload.end());

    const std::uint32_t total_size = static_cast<std::uint32_t>(container.size());
    container[8] = static_cast<std::uint8_t>(total_size & 0xFF);
    container[9] = static_cast<std::uint8_t>((total_size >> 8) & 0xFF);
    container[10] = static_cast<std::uint8_t>((total_size >> 16) & 0xFF);
    container[11] = static_cast<std::uint8_t>((total_size >> 24) & 0xFF);

    if (!cumetal::common::write_file_bytes(options.output, container, &io_error)) {
        result.error = io_error;
        return result;
    }

    result.output = options.output;
    result.logs.push_back("wrote experimental container with " + std::to_string(payload.size()) +
                          " bytes of payload");
    result.ok = true;
    return result;
}

EmitResult emit_with_xcrun(const EmitOptions& options) {
    EmitResult result;
    result.mode_used = EmitMode::kXcrun;

    if (!std::filesystem::exists(options.input)) {
        result.error = "input file does not exist: " + options.input.string();
        return result;
    }

    const std::filesystem::path temp_dir = make_temp_dir();
    const std::filesystem::path temp_air = temp_dir / "kernel.air";

    const std::string extension = lower_ext(options.input);
    bool prepared_air = false;

    if (extension == ".air") {
        std::filesystem::copy_file(options.input, temp_air,
                                   std::filesystem::copy_options::overwrite_existing);
        prepared_air = true;
        result.logs.push_back("using provided .air payload");
    } else if (extension == ".metal") {
        const std::string command = "xcrun metal -c " + quote_shell(options.input.string()) +
                                    " -o " + quote_shell(temp_air.string()) + " 2>&1";
        const CommandResult cmd = run_command_capture(command);
        result.logs.push_back("$ " + command);
        result.logs.push_back(cmd.output);
        if (!cmd.started || cmd.exit_code != 0) {
            result.error = "failed to compile .metal to .air with xcrun metal";
            std::filesystem::remove_all(temp_dir);
            return result;
        }
        prepared_air = true;
    } else if (extension == ".ll" || extension == ".llvm") {
        const std::string command = "llvm-as " + quote_shell(options.input.string()) + " -o " +
                                    quote_shell(temp_air.string()) + " 2>&1";
        const CommandResult cmd = run_command_capture(command);
        result.logs.push_back("$ " + command);
        result.logs.push_back(cmd.output);
        if (!cmd.started || cmd.exit_code != 0) {
            result.error = "failed to assemble LLVM IR with llvm-as";
            std::filesystem::remove_all(temp_dir);
            return result;
        }
        prepared_air = true;
    } else if (extension == ".bc") {
        std::filesystem::copy_file(options.input, temp_air,
                                   std::filesystem::copy_options::overwrite_existing);
        prepared_air = true;
        result.logs.push_back("treating input .bc as AIR payload for metallib packaging");
    } else {
        result.error = "unsupported input extension for xcrun mode: " + extension;
        std::filesystem::remove_all(temp_dir);
        return result;
    }

    if (!prepared_air) {
        result.error = "could not prepare AIR payload for metallib packaging";
        std::filesystem::remove_all(temp_dir);
        return result;
    }

    const std::string metallib_command = "xcrun metallib " + quote_shell(temp_air.string()) +
                                         " -o " + quote_shell(options.output.string()) + " 2>&1";
    const CommandResult pack_cmd = run_command_capture(metallib_command);
    result.logs.push_back("$ " + metallib_command);
    result.logs.push_back(pack_cmd.output);

    std::filesystem::remove_all(temp_dir);

    if (!pack_cmd.started || pack_cmd.exit_code != 0) {
        result.error = "failed to package metallib with xcrun metallib";
        return result;
    }

    result.ok = true;
    result.output = options.output;
    return result;
}

EmitResult validate_if_requested(const EmitOptions& options, EmitResult result) {
    if (!result.ok || !options.validate_output) {
        return result;
    }

    cumetal::air_validate::ValidationOptions validation_options;
    validation_options.require_bitcode = (result.mode_used == EmitMode::kXcrun);
    validation_options.strict_magic = true;
    validation_options.run_xcrun_validate = options.run_xcrun_validate;

    const auto validation = cumetal::air_validate::validate_file(options.output, validation_options);
    result.logs.push_back(cumetal::air_validate::format_report(validation));
    if (!validation.ok) {
        result.ok = false;
        result.error = "output metallib did not pass validation";
    }
    return result;
}

}  // namespace

EmitResult emit_metallib(const EmitOptions& options) {
    EmitResult result;

    if (options.input.empty() || options.output.empty()) {
        result.error = "both input and output paths are required";
        return result;
    }

    if (std::filesystem::exists(options.output) && !options.overwrite) {
        result.error = "output already exists (pass --overwrite to replace): " +
                       options.output.string();
        return result;
    }

    if (const auto parent = options.output.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    if (options.mode == EmitMode::kXcrun) {
        EmitResult xcrun_result = emit_with_xcrun(options);
        if (!xcrun_result.ok && options.fallback_to_experimental) {
            EmitResult fallback_result = emit_experimental(options);
            fallback_result.logs.insert(fallback_result.logs.begin(),
                                        "xcrun mode failed: " + xcrun_result.error);
            fallback_result.logs.insert(fallback_result.logs.begin(),
                                        "xcrun mode failed, falling back to experimental container mode");
            fallback_result.logs.insert(fallback_result.logs.begin(), xcrun_result.logs.begin(),
                                        xcrun_result.logs.end());
            result = std::move(fallback_result);
        } else {
            result = std::move(xcrun_result);
        }
    } else {
        result = emit_experimental(options);
    }

    return validate_if_requested(options, std::move(result));
}

}  // namespace cumetal::air_emitter
