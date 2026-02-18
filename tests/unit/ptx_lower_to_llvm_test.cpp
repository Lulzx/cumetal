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
    if (!expect(contains(lowered.llvm_ir, "float addrspace(1)* %vector_add_param_0"),
                "u64 param mapped")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "i32 %vector_add_param_3"), "u32 param mapped")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "%sum = fadd float %a.val, %b.val"),
                "vector-add floating add body emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "store float %sum, float addrspace(1)* %c.ptr"),
                "vector-add store emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "\"air.kernel\""), "air.kernel attribute emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "\"air.version\"=\"2.8\""), "air.version emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "!air.language_version = !{!"),
                "air language version metadata emitted")) {
        return 1;
    }
    if (!expect(lowered.warnings.empty(), "no warnings for supported vector-add lowering path")) {
        return 1;
    }

    const std::string matrix_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry matrix_mul(
    .param .u64 matrix_mul_param_0,
    .param .u64 matrix_mul_param_1,
    .param .u64 matrix_mul_param_2,
    .param .u32 matrix_mul_param_3,
    .param .u32 matrix_mul_param_4
)
{
    mov.u32 %r0, %tid.x;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions matrix_options;
    matrix_options.entry_name = "matrix_mul";
    matrix_options.module_id = "unit.ptx.matrix_mul";
    const auto matrix_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(matrix_ptx, matrix_options);
    if (!expect(matrix_lowered.ok, "matrix multiply lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir,
                         "define void @matrix_mul(float addrspace(1)* %matrix_mul_param_0"),
                "matrix multiply kernel definition emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir,
                         "%row = udiv i32 %matrix_mul_param_4, %n.val"),
                "matrix row index derivation emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir, "%prod = fmul float %a.val, %b.val"),
                "matrix multiply fmul emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir, "%acc.next = fadd float %acc, %prod"),
                "matrix multiply accumulation emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir, "store float %acc, float addrspace(1)* %c.ptr"),
                "matrix multiply store emitted")) {
        return 1;
    }
    if (!expect(matrix_lowered.warnings.empty(), "no warnings for matrix multiply lowering path")) {
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
