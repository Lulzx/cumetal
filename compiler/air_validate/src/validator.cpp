#include "cumetal/air_validate/validator.h"

#include <cstdio>
#include <sstream>
#include <sys/wait.h>

namespace cumetal::air_validate {
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
    } else {
        result.exit_code = -1;
    }

    return result;
}

bool has_error(const std::vector<Diagnostic>& diagnostics) {
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.severity == Severity::kError) {
            return true;
        }
    }
    return false;
}

const char* severity_to_string(Severity severity) {
    switch (severity) {
        case Severity::kInfo:
            return "INFO";
        case Severity::kWarning:
            return "WARN";
        case Severity::kError:
            return "ERROR";
    }
    return "UNKNOWN";
}

}  // namespace

ValidationResult validate_file(const std::filesystem::path& path, const ValidationOptions& options) {
    ValidationResult result;

    std::string io_error;
    const std::vector<std::uint8_t> bytes = cumetal::common::read_file_bytes(path, &io_error);
    if (bytes.empty()) {
        result.diagnostics.push_back(
            {.severity = Severity::kError, .message = io_error.empty() ? "input file is empty" : io_error});
        result.ok = false;
        return result;
    }

    result.summary =
        cumetal::common::inspect_metallib_bytes(path.string(), bytes, options.max_strings);

    if (result.summary.file_size < 16) {
        result.diagnostics.push_back(
            {.severity = Severity::kError,
             .message = "container is too small to be a valid metallib (expected >= 16 bytes)"});
    }

    if (options.strict_magic && !cumetal::common::looks_like_metallib(result.summary)) {
        result.diagnostics.push_back(
            {.severity = Severity::kError,
             .message = "magic header does not match expected MTL* container prefix"});
    }

    if (options.require_bitcode && result.summary.bitcode_sections.empty()) {
        result.diagnostics.push_back(
            {.severity = Severity::kError,
             .message = "no embedded LLVM bitcode signatures found in container"});
    } else if (!result.summary.bitcode_sections.empty()) {
        result.diagnostics.push_back(
            {.severity = Severity::kInfo,
             .message = "found " + std::to_string(result.summary.bitcode_sections.size()) +
                        " candidate embedded bitcode section(s)"});
    }

    if (options.run_xcrun_validate) {
        const std::string command =
            "xcrun metal -validate " + quote_shell(path.string()) + " 2>&1";
        const CommandResult command_result = run_command_capture(command);
        result.xcrun_exit_code = command_result.exit_code;
        result.xcrun_output = command_result.output;

        if (!command_result.started) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "failed to invoke xcrun for metal -validate"});
        } else if (command_result.exit_code != 0) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "xcrun metal -validate returned non-zero exit code " +
                            std::to_string(command_result.exit_code)});
        } else {
            result.diagnostics.push_back(
                {.severity = Severity::kInfo,
                 .message = "xcrun metal -validate accepted the container"});
        }
    }

    result.ok = !has_error(result.diagnostics);
    return result;
}

std::string format_report(const ValidationResult& result) {
    std::ostringstream out;

    out << "Validation result: " << (result.ok ? "PASS" : "FAIL") << "\n";
    out << "File: " << result.summary.path << "\n";
    out << "Size: " << result.summary.file_size << " bytes\n";
    out << "Magic: " << result.summary.magic_ascii << "\n";
    out << "Bitcode sections: " << result.summary.bitcode_sections.size() << "\n";

    for (const auto& diagnostic : result.diagnostics) {
        out << "[" << severity_to_string(diagnostic.severity) << "] " << diagnostic.message << "\n";
    }

    if (!result.xcrun_output.empty()) {
        out << "\n--- xcrun output ---\n";
        out << result.xcrun_output;
        if (result.xcrun_output.back() != '\n') {
            out << "\n";
        }
    }

    return out.str();
}

}  // namespace cumetal::air_validate
