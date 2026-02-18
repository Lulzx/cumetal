#pragma once

#include "cumetal/passes/addrspace.h"
#include "cumetal/passes/intrinsic_lower.h"
#include "cumetal/passes/metadata.h"
#include "cumetal/passes/printf_lower.h"

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::passes {

struct Phase1PipelineOptions {
    bool strict = false;
    std::string entry_name;
    MetadataOptions metadata;
};

struct Phase1PipelineOutput {
    bool ok = false;
    std::string entry_name;
    std::vector<LoweredInstruction> lowered_instructions;
    std::vector<PrintfLoweredCall> printf_calls;
    std::vector<PrintfFormatEntry> printf_formats;
    std::vector<AddrspaceInstruction> addrspace_instructions;
    KernelMetadata metadata;
    std::vector<std::string> warnings;
    std::string error;
};

Phase1PipelineOutput run_phase1_pipeline(std::string_view ptx,
                                         const Phase1PipelineOptions& options = {});

}  // namespace cumetal::passes
