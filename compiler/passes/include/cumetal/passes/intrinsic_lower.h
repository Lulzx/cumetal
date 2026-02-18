#pragma once

#include "cumetal/ptx/parser.h"

#include <string>
#include <vector>

namespace cumetal::passes {

struct LoweredInstruction {
    std::string opcode;
    std::vector<std::string> operands;
    bool translated = false;
};

struct IntrinsicLowerResult {
    bool ok = false;
    std::vector<LoweredInstruction> instructions;
    std::vector<std::string> warnings;
    std::string error;
};

struct IntrinsicLowerOptions {
    bool strict = false;
};

IntrinsicLowerResult lower_intrinsics(const cumetal::ptx::EntryFunction& entry,
                                      const IntrinsicLowerOptions& options = {});

}  // namespace cumetal::passes
