#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cumetal::air_emitter {

enum class EmitMode {
    kXcrun,
    kExperimentalContainer,
};

struct EmitOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    EmitMode mode = EmitMode::kXcrun;
    bool overwrite = false;
    bool fallback_to_experimental = false;
    bool validate_output = true;
    bool run_xcrun_validate = false;
    std::string kernel_name = "vector_add";
};

struct EmitResult {
    bool ok = false;
    EmitMode mode_used = EmitMode::kXcrun;
    std::filesystem::path output;
    std::vector<std::string> logs;
    std::string error;
};

EmitResult emit_metallib(const EmitOptions& options);

}  // namespace cumetal::air_emitter
