#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

struct LowerToMetalOptions {
    bool strict = false;
    std::string entry_name;
};

struct LowerToMetalResult {
    bool ok = false;
    bool matched = false;
    std::string entry_name;
    std::string metal_source;
    // printf metadata: if the kernel uses device printf, the compiler injects a hidden
    // ring-buffer argument (spec §5.3).  printf_formats[i] is the format string for id i.
    std::vector<std::string> printf_formats;
    std::vector<std::string> warnings;
    std::string error;
};

LowerToMetalResult lower_ptx_to_metal_source(std::string_view ptx,
                                             const LowerToMetalOptions& options = {});

}  // namespace cumetal::ptx
