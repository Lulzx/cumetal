#pragma once

#include "cumetal/common/metallib.h"

#include <filesystem>
#include <string>
#include <vector>

namespace cumetal::air_validate {

enum class Severity {
    kInfo,
    kWarning,
    kError,
};

struct Diagnostic {
    Severity severity = Severity::kInfo;
    std::string message;
};

struct ValidationOptions {
    bool require_bitcode = true;
    bool run_xcrun_validate = false;
    bool strict_magic = true;
    std::size_t max_strings = 32;
};

struct ValidationResult {
    bool ok = false;
    int xcrun_exit_code = -1;
    std::string xcrun_output;
    cumetal::common::MetallibSummary summary;
    std::vector<Diagnostic> diagnostics;
};

ValidationResult validate_file(const std::filesystem::path& path, const ValidationOptions& options);
std::string format_report(const ValidationResult& result);

}  // namespace cumetal::air_validate
