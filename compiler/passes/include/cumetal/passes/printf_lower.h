#pragma once

#include "cumetal/ptx/parser.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cumetal::passes {

struct PrintfFormatEntry {
    std::uint32_t id = 0;
    std::string token;
    bool literal = false;
    bool truncated = false;
};

struct PrintfLoweredCall {
    int source_line = 0;
    std::string source_opcode;
    std::uint32_t format_id = 0;
    std::string format_token;
    std::vector<std::string> arguments;
};

struct PrintfLowerOptions {
    bool strict = false;
    std::size_t max_format_length = 256;
};

struct PrintfLowerResult {
    bool ok = false;
    std::vector<PrintfLoweredCall> calls;
    std::vector<PrintfFormatEntry> formats;
    std::vector<std::string> warnings;
    std::string error;
};

PrintfLowerResult lower_printf_calls(const cumetal::ptx::EntryFunction& entry,
                                     const PrintfLowerOptions& options = {});

}  // namespace cumetal::passes
