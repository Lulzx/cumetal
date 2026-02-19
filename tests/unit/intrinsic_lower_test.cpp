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
    div.s32 %r7, %r3, %r0;
    rem.s32 %r8, %r3, %r0;
    and.b32 %r9, %r7, %r8;
    or.b32  %r10, %r7, %r8;
    xor.b32 %r11, %r9, %r10;
    not.b32 %r12, %r11;
    selp.f32 %f2, %f0, %f1, %p1;
    rcp.rn.f32 %f3, %f2;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    bar.sync 0;
    call.uni (%r6), vprintf, ("tid=%u", %r0);
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
    if (!expect(lowered.instructions[4].translated && lowered.instructions[4].opcode == "llvm.div",
                "div mapped to llvm.div")) {
        return 1;
    }
    if (!expect(lowered.instructions[5].translated && lowered.instructions[5].opcode == "llvm.rem",
                "rem mapped to llvm.rem")) {
        return 1;
    }
    if (!expect(lowered.instructions[6].translated && lowered.instructions[6].opcode == "llvm.and",
                "and mapped to llvm.and")) {
        return 1;
    }
    if (!expect(lowered.instructions[7].translated && lowered.instructions[7].opcode == "llvm.or",
                "or mapped to llvm.or")) {
        return 1;
    }
    if (!expect(lowered.instructions[8].translated && lowered.instructions[8].opcode == "llvm.xor",
                "xor mapped to llvm.xor")) {
        return 1;
    }
    if (!expect(lowered.instructions[9].translated && lowered.instructions[9].opcode == "llvm.not",
                "not mapped to llvm.not")) {
        return 1;
    }
    if (!expect(lowered.instructions[10].translated && lowered.instructions[10].opcode == "llvm.select",
                "selp mapped to llvm.select")) {
        return 1;
    }
    if (!expect(lowered.instructions[11].translated && lowered.instructions[11].opcode == "llvm.rcp",
                "rcp mapped to llvm.rcp")) {
        return 1;
    }
    if (!expect(lowered.instructions[12].translated && lowered.instructions[12].opcode == "llvm.mad",
                "integer mad mapped to llvm.mad")) {
        return 1;
    }
    if (!expect(lowered.instructions[13].translated &&
                    lowered.instructions[13].opcode == "air.threadgroup_barrier",
                "bar.sync mapped to barrier intrinsic")) {
        return 1;
    }
    if (!expect(!lowered.instructions[14].translated &&
                    lowered.instructions[14].opcode == "call.uni",
                "call instruction preserved for printf lowering stage")) {
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

    // ── Test 2: math intrinsics (max, min, abs, fma, sqrt, rsqrt, ex2, lg2, sin, cos) ──
    const std::string ptx2 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k2(
    .param .u64 p0,
    .param .u64 p1
)
{
    max.f32    %f0, %f1, %f2;
    min.f32    %f0, %f1, %f2;
    abs.f32    %f0, %f1;
    fma.rn.f32 %f0, %f1, %f2, %f3;
    sqrt.rn.f32      %f0, %f1;
    rsqrt.approx.f32 %f0, %f1;
    ex2.approx.f32   %f0, %f1;
    lg2.approx.f32   %f0, %f1;
    sin.approx.f32   %f0, %f1;
    cos.approx.f32   %f0, %f1;
    ret;
}
)PTX";

    const auto parsed2 = cumetal::ptx::parse_ptx(ptx2);
    if (!expect(parsed2.ok, "parse PTX2 for math intrinsics")) return 1;
    if (!expect(parsed2.module.entries.size() == 1, "single PTX2 entry")) return 1;
    if (!expect(parsed2.module.entries[0].instructions.size() == 11,
                "PTX2 instruction count (10 math + ret)"))
        return 1;

    const auto m = cumetal::passes::lower_intrinsics(parsed2.module.entries[0]);
    if (!expect(m.ok, "math intrinsics lower ok")) return 1;
    if (!expect(m.warnings.empty(), "no warnings for supported math intrinsics")) return 1;

    if (!expect(m.instructions[0].translated && m.instructions[0].opcode == "llvm.fmax",
                "max.f32 → llvm.fmax"))
        return 1;
    if (!expect(m.instructions[1].translated && m.instructions[1].opcode == "llvm.fmin",
                "min.f32 → llvm.fmin"))
        return 1;
    if (!expect(m.instructions[2].translated && m.instructions[2].opcode == "llvm.fabs",
                "abs.f32 → llvm.fabs"))
        return 1;
    if (!expect(m.instructions[3].translated && m.instructions[3].opcode == "llvm.fma",
                "fma.rn.f32 → llvm.fma"))
        return 1;
    if (!expect(m.instructions[4].translated && m.instructions[4].opcode == "llvm.sqrt",
                "sqrt.rn.f32 → llvm.sqrt"))
        return 1;
    if (!expect(m.instructions[5].translated && m.instructions[5].opcode == "llvm.rsqrt",
                "rsqrt.approx.f32 → llvm.rsqrt"))
        return 1;
    if (!expect(m.instructions[6].translated && m.instructions[6].opcode == "llvm.exp2",
                "ex2.approx.f32 → llvm.exp2"))
        return 1;
    if (!expect(m.instructions[7].translated && m.instructions[7].opcode == "llvm.log2",
                "lg2.approx.f32 → llvm.log2"))
        return 1;
    if (!expect(m.instructions[8].translated && m.instructions[8].opcode == "llvm.sin",
                "sin.approx.f32 → llvm.sin"))
        return 1;
    if (!expect(m.instructions[9].translated && m.instructions[9].opcode == "llvm.cos",
                "cos.approx.f32 → llvm.cos"))
        return 1;

    // ── Test 3: warp primitives (shfl, vote, bar.warp.sync) ──────────────────
    const std::string ptx3 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k3(
    .param .u64 p0
)
{
    shfl.sync.idx.b32   %r0, %r1, %r2, 0x1f, 0xffffffff;
    shfl.sync.down.b32  %r0, %r1, %r2, 0x1f, 0xffffffff;
    shfl.sync.up.b32    %r0, %r1, %r2, 0x1f, 0xffffffff;
    shfl.sync.bfly.b32  %r0, %r1, %r2, 0x1f, 0xffffffff;
    vote.sync.ballot.b32 %r0, %p0, 0xffffffff;
    vote.sync.any.pred  %p0, %p1, 0xffffffff;
    vote.sync.all.pred  %p0, %p1, 0xffffffff;
    bar.warp.sync       0xffffffff;
    ret;
}
)PTX";

    const auto parsed3 = cumetal::ptx::parse_ptx(ptx3);
    if (!expect(parsed3.ok, "parse PTX3 for warp primitives")) return 1;
    if (!expect(parsed3.module.entries.size() == 1, "single PTX3 entry")) return 1;
    if (!expect(parsed3.module.entries[0].instructions.size() == 9,
                "PTX3 instruction count (8 warp ops + ret)"))
        return 1;

    const auto w = cumetal::passes::lower_intrinsics(parsed3.module.entries[0]);
    if (!expect(w.ok, "warp primitives lower ok")) return 1;
    if (!expect(w.warnings.empty(), "no warnings for supported warp primitives")) return 1;

    if (!expect(w.instructions[0].translated && w.instructions[0].opcode == "air.simdgroup.shuffle",
                "shfl.sync.idx → air.simdgroup.shuffle"))
        return 1;
    if (!expect(w.instructions[1].translated && w.instructions[1].opcode == "air.simdgroup.shuffle_down",
                "shfl.sync.down → air.simdgroup.shuffle_down"))
        return 1;
    if (!expect(w.instructions[2].translated && w.instructions[2].opcode == "air.simdgroup.shuffle_up",
                "shfl.sync.up → air.simdgroup.shuffle_up"))
        return 1;
    if (!expect(w.instructions[3].translated && w.instructions[3].opcode == "air.simdgroup.shuffle_xor",
                "shfl.sync.bfly → air.simdgroup.shuffle_xor"))
        return 1;
    if (!expect(w.instructions[4].translated && w.instructions[4].opcode == "air.simdgroup.ballot",
                "vote.sync.ballot → air.simdgroup.ballot"))
        return 1;
    if (!expect(w.instructions[5].translated && w.instructions[5].opcode == "air.simdgroup.any",
                "vote.sync.any → air.simdgroup.any"))
        return 1;
    if (!expect(w.instructions[6].translated && w.instructions[6].opcode == "air.simdgroup.all",
                "vote.sync.all → air.simdgroup.all"))
        return 1;
    if (!expect(w.instructions[7].translated && w.instructions[7].opcode == "air.simdgroup.barrier",
                "bar.warp.sync → air.simdgroup.barrier"))
        return 1;

    std::printf("PASS: intrinsic lower unit tests\n");
    return 0;
}
