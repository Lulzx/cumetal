#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

struct Parameter {
    std::string type;
    std::string name;
};

struct EntryFunction {
    std::string name;
    std::vector<Parameter> params;
};

struct ModuleInfo {
    int version_major = -1;
    int version_minor = -1;
    std::string target;
    std::vector<EntryFunction> entries;
};

struct ParseResult {
    bool ok = false;
    ModuleInfo module;
    std::string error;
};

ParseResult parse_ptx(std::string_view text);

}  // namespace cumetal::ptx
