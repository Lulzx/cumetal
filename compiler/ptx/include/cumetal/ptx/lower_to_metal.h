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
    std::vector<std::string> warnings;
    std::string error;
};

LowerToMetalResult lower_ptx_to_metal_source(std::string_view ptx,
                                             const LowerToMetalOptions& options = {});

}  // namespace cumetal::ptx
