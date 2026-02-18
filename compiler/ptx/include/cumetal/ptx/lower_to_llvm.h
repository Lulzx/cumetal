#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

struct LowerToLlvmOptions {
    bool strict = false;
    std::string entry_name;
    std::string module_id = "cumetal.ptx.module";
    std::string target_triple = "air64-apple-macosx14.0.0";
};

struct LowerToLlvmResult {
    bool ok = false;
    std::string entry_name;
    std::string llvm_ir;
    std::vector<std::string> warnings;
    std::string error;
};

LowerToLlvmResult lower_ptx_to_llvm_ir(std::string_view ptx,
                                       const LowerToLlvmOptions& options = {});

}  // namespace cumetal::ptx
