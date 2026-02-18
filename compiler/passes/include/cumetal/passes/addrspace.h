#pragma once

#include "cumetal/ptx/parser.h"

#include <string>
#include <vector>

namespace cumetal::passes {

struct AddrspaceInstruction {
    std::string opcode;
    std::vector<std::string> operands;
    int address_space = 0;
    bool rewritten = false;
};

struct AddrspaceRewriteOptions {
    bool strict = false;
};

struct AddrspaceRewriteResult {
    bool ok = false;
    std::vector<AddrspaceInstruction> instructions;
    std::vector<std::string> warnings;
    std::string error;
};

AddrspaceRewriteResult rewrite_addrspace(const cumetal::ptx::EntryFunction& entry,
                                         const AddrspaceRewriteOptions& options = {});

}  // namespace cumetal::passes
