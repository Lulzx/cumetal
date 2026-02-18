#include "cumetal/passes/intrinsic_lower.h"
#include "cumetal/ptx/parser.h"

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

}  // namespace

int main() {
    const std::string ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry k(
    .param .u64 p0,
    .param .u64 p1
)
{
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.y;
    add.s32 %r2, %r0, %r1;
    mul.lo.s32 %r3, %r2, %r0;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    bar.sync 0;
    foo.bar %r5, %r4;
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(ptx);
    if (!expect(parsed.ok, "parse PTX for intrinsic lowering")) {
        return 1;
    }
    if (!expect(parsed.module.entries.size() == 1, "single PTX entry")) {
        return 1;
    }

    const auto lowered = cumetal::passes::lower_intrinsics(parsed.module.entries[0]);
    if (!expect(lowered.ok, "non-strict intrinsic lower succeeds")) {
        return 1;
    }
    if (!expect(lowered.instructions.size() == parsed.module.entries[0].instructions.size(),
                "instruction count preserved")) {
        return 1;
    }
    if (!expect(lowered.instructions[0].translated, "tid.x mov translated")) {
        return 1;
    }
    if (!expect(lowered.instructions[0].opcode == "air.read.special_register",
                "tid.x mapped to special-register intrinsic")) {
        return 1;
    }
    if (!expect(lowered.instructions[0].operands.size() == 2 &&
                    lowered.instructions[0].operands[1] == "air.thread_position_in_threadgroup.x",
                "tid.x mapping operand")) {
        return 1;
    }
    if (!expect(lowered.instructions[1].translated &&
                    lowered.instructions[1].operands[1] == "air.threadgroup_position_in_grid.y",
                "ctaid.y mapping")) {
        return 1;
    }
    if (!expect(lowered.instructions[2].translated && lowered.instructions[2].opcode == "llvm.add",
                "add mapped to llvm.add")) {
        return 1;
    }
    if (!expect(lowered.instructions[3].translated && lowered.instructions[3].opcode == "llvm.mul",
                "mul mapped to llvm.mul")) {
        return 1;
    }
    if (!expect(lowered.instructions[4].translated && lowered.instructions[4].opcode == "llvm.mad",
                "integer mad mapped to llvm.mad")) {
        return 1;
    }
    if (!expect(lowered.instructions[5].translated &&
                    lowered.instructions[5].opcode == "air.threadgroup_barrier",
                "bar.sync mapped to barrier intrinsic")) {
        return 1;
    }
    if (!expect(!lowered.warnings.empty(), "unsupported opcode emits warning")) {
        return 1;
    }

    cumetal::passes::IntrinsicLowerOptions strict_options;
    strict_options.strict = true;
    const auto strict = cumetal::passes::lower_intrinsics(parsed.module.entries[0], strict_options);
    if (!expect(!strict.ok, "strict intrinsic lowering fails on unmapped opcode")) {
        return 1;
    }
    if (!expect(strict.error.find("no mapping for opcode") != std::string::npos,
                "strict error message includes unmapped opcode")) {
        return 1;
    }

    std::printf("PASS: intrinsic lower unit tests\n");
    return 0;
}
