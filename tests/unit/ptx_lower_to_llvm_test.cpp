#include "cumetal/ptx/lower_to_llvm.h"

#include <cstdio>
#include <string>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
    const std::string ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u32 vector_add_param_3
)
{
    mov.u32 %r0, %tid.x;
    add.s32 %r1, %r0, %r0;
    ld.shared.u32 %r2, [%rd1];
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions options;
    options.entry_name = "vector_add";
    options.module_id = "unit.ptx.vector_add";
    const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx, options);
    if (!expect(lowered.ok, "lower_ptx_to_llvm_ir succeeds")) {
        return 1;
    }
    if (!expect(lowered.entry_name == "vector_add", "entry name propagated")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "; ModuleID = 'unit.ptx.vector_add'"), "module id emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "define void @vector_add("), "kernel definition emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "ptr addrspace(1) %vector_add_param_0"), "u64 param mapped")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "i32 %vector_add_param_3"), "u32 param mapped")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "%sum = fadd float %a.val, %b.val"),
                "vector-add floating add body emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "store float %sum, ptr addrspace(1) %c.ptr"),
                "vector-add store emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "\"air.kernel\""), "air.kernel attribute emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "\"air.version\"=\"2.8\""), "air.version emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "!air.language_version = !{!2}"),
                "air language version metadata emitted")) {
        return 1;
    }
    if (!expect(lowered.warnings.empty(), "no warnings for supported vector-add lowering path")) {
        return 1;
    }

    const std::string unsupported_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u32 vector_add_param_3
)
{
    foo.shared.u32 %r3, %r2;
    ret;
}
)PTX";

    const auto tolerant = cumetal::ptx::lower_ptx_to_llvm_ir(unsupported_ptx, options);
    if (!expect(tolerant.ok, "tolerant lowering accepts unsupported opcode")) {
        return 1;
    }
    if (!expect(!tolerant.warnings.empty(), "warnings propagated for unsupported opcode")) {
        return 1;
    }

    cumetal::ptx::LowerToLlvmOptions strict_options;
    strict_options.entry_name = "vector_add";
    strict_options.strict = true;
    const auto strict = cumetal::ptx::lower_ptx_to_llvm_ir(unsupported_ptx, strict_options);
    if (!expect(!strict.ok, "strict lowering fails on unsupported opcode set")) {
        return 1;
    }

    std::printf("PASS: ptx lower-to-llvm unit tests\n");
    return 0;
}
