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

    // ── Test 4: membar, cp.async, redux.sync ────────────────────────────────
    const std::string ptx4 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k4(
    .param .u64 p0,
    .param .u64 p1
)
{
    membar.gl;
    membar.cta;
    cp.async.ca.shared.global [p0], [p1], 16;
    cp.async.commit_group;
    cp.async.wait_group 0;
    redux.sync.add.s32 %r0, %r1, 0xffffffff;
    redux.sync.add.f32 %f0, %f1, 0xffffffff;
    ret;
}
)PTX";

    const auto parsed4 = cumetal::ptx::parse_ptx(ptx4);
    if (!expect(parsed4.ok, "parse PTX4 for membar/cp.async/redux")) return 1;
    if (!expect(parsed4.module.entries.size() == 1, "single PTX4 entry")) return 1;
    if (!expect(parsed4.module.entries[0].instructions.size() == 8,
                "PTX4 instruction count (7 ops + ret)"))
        return 1;

    const auto z = cumetal::passes::lower_intrinsics(parsed4.module.entries[0]);
    if (!expect(z.ok, "membar/cp.async/redux lower ok")) return 1;
    if (!expect(z.warnings.empty(), "no warnings for supported ops")) return 1;

    if (!expect(z.instructions[0].translated && z.instructions[0].opcode == "air.mem.barrier.device",
                "membar.gl → air.mem.barrier.device"))
        return 1;
    if (!expect(z.instructions[1].translated && z.instructions[1].opcode == "air.mem.barrier.threadgroup",
                "membar.cta → air.mem.barrier.threadgroup"))
        return 1;
    if (!expect(z.instructions[2].translated && z.instructions[2].opcode == "air.cp_async",
                "cp.async → air.cp_async"))
        return 1;
    if (!expect(z.instructions[3].translated && z.instructions[3].opcode == "air.threadgroup_barrier",
                "cp.async.commit_group → air.threadgroup_barrier"))
        return 1;
    if (!expect(z.instructions[4].translated && z.instructions[4].opcode == "air.threadgroup_barrier",
                "cp.async.wait_group → air.threadgroup_barrier"))
        return 1;
    if (!expect(z.instructions[5].translated && z.instructions[5].opcode == "air.simdgroup.reduce_add",
                "redux.sync.add.s32 → air.simdgroup.reduce_add"))
        return 1;
    if (!expect(z.instructions[6].translated &&
                    z.instructions[6].opcode == "air.simdgroup.reduce_add.f32",
                "redux.sync.add.f32 → air.simdgroup.reduce_add.f32"))
        return 1;

    // ── Test 5: clz and popc (count-leading-zeros, population-count) ────────
    const std::string ptx5 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k5(
    .param .u64 p0,
    .param .u64 p1
)
{
    clz.b32  %r0, %r1;
    clz.b64  %r0, %rd1;
    popc.b32 %r0, %r1;
    popc.b64 %r0, %rd1;
    ret;
}
)PTX";

    const auto parsed5 = cumetal::ptx::parse_ptx(ptx5);
    if (!expect(parsed5.ok, "parse PTX5 for clz/popc")) return 1;
    if (!expect(parsed5.module.entries.size() == 1, "single PTX5 entry")) return 1;
    if (!expect(parsed5.module.entries[0].instructions.size() == 5,
                "PTX5 instruction count (4 bit ops + ret)"))
        return 1;

    const auto bp = cumetal::passes::lower_intrinsics(parsed5.module.entries[0]);
    if (!expect(bp.ok, "clz/popc lower ok")) return 1;
    if (!expect(bp.warnings.empty(), "no warnings for clz/popc")) return 1;

    if (!expect(bp.instructions[0].translated && bp.instructions[0].opcode == "llvm.ctlz.i32",
                "clz.b32 → llvm.ctlz.i32"))
        return 1;
    if (!expect(bp.instructions[1].translated && bp.instructions[1].opcode == "llvm.ctlz.i64",
                "clz.b64 → llvm.ctlz.i64"))
        return 1;
    if (!expect(bp.instructions[2].translated && bp.instructions[2].opcode == "llvm.ctpop.i32",
                "popc.b32 → llvm.ctpop.i32"))
        return 1;
    if (!expect(bp.instructions[3].translated && bp.instructions[3].opcode == "llvm.ctpop.i64",
                "popc.b64 → llvm.ctpop.i64"))
        return 1;

    // ── Test 6: abs (integer/float) and shr (logical/arithmetic) ────────────
    const std::string ptx6 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k6(.param .u64 p0, .param .u64 p1)
{
    abs.s32   %r0, %r1;
    abs.s64   %rd0, %rd1;
    abs.f32   %f0, %f1;
    abs.f64   %fd0, %fd1;
    shr.b32   %r2, %r1, 3;
    shr.s32   %r3, %r1, 2;
    shr.u32   %r4, %r1, 5;
    shr.b64   %rd2, %rd1, 4;
    ret;
})PTX";

    const auto parsed6 = cumetal::ptx::parse_ptx(ptx6);
    if (!expect(parsed6.ok, "parse PTX6 for abs/shr")) return 1;
    if (!expect(parsed6.module.entries.size() == 1, "single PTX6 entry")) return 1;
    if (!expect(parsed6.module.entries[0].instructions.size() == 9,
                "PTX6 instruction count (8 ops + ret)"))
        return 1;

    const auto t6 = cumetal::passes::lower_intrinsics(parsed6.module.entries[0]);
    if (!expect(t6.ok, "abs/shr lower ok")) return 1;
    if (!expect(t6.warnings.empty(), "no warnings for abs/shr")) return 1;

    if (!expect(t6.instructions[0].translated && t6.instructions[0].opcode == "llvm.abs",
                "abs.s32 → llvm.abs"))
        return 1;
    if (!expect(t6.instructions[1].translated && t6.instructions[1].opcode == "llvm.abs",
                "abs.s64 → llvm.abs"))
        return 1;
    if (!expect(t6.instructions[2].translated && t6.instructions[2].opcode == "llvm.fabs",
                "abs.f32 → llvm.fabs"))
        return 1;
    if (!expect(t6.instructions[3].translated && t6.instructions[3].opcode == "llvm.fabs",
                "abs.f64 → llvm.fabs"))
        return 1;
    if (!expect(t6.instructions[4].translated && t6.instructions[4].opcode == "llvm.shr",
                "shr.b32 → llvm.shr"))
        return 1;
    if (!expect(t6.instructions[5].translated && t6.instructions[5].opcode == "llvm.shr",
                "shr.s32 → llvm.shr"))
        return 1;
    if (!expect(t6.instructions[6].translated && t6.instructions[6].opcode == "llvm.shr",
                "shr.u32 → llvm.shr"))
        return 1;
    if (!expect(t6.instructions[7].translated && t6.instructions[7].opcode == "llvm.shr",
                "shr.b64 → llvm.shr"))
        return 1;

    // ── Test 7: brev (bit reverse) ───────────────────────────────────────────
    const std::string ptx7 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k7(.param .u64 p0, .param .u64 p1)
{
    brev.b32 %r0, %r1;
    brev.b64 %rd0, %rd1;
    ret;
})PTX";

    const auto parsed7 = cumetal::ptx::parse_ptx(ptx7);
    if (!expect(parsed7.ok, "parse PTX7 for brev")) return 1;
    if (!expect(parsed7.module.entries.size() == 1, "single PTX7 entry")) return 1;
    if (!expect(parsed7.module.entries[0].instructions.size() == 3,
                "PTX7 instruction count (2 brev ops + ret)"))
        return 1;

    const auto t7 = cumetal::passes::lower_intrinsics(parsed7.module.entries[0]);
    if (!expect(t7.ok, "brev lower ok")) return 1;
    if (!expect(t7.warnings.empty(), "no warnings for brev")) return 1;

    if (!expect(t7.instructions[0].translated &&
                    t7.instructions[0].opcode == "llvm.bitreverse.i32",
                "brev.b32 → llvm.bitreverse.i32"))
        return 1;
    if (!expect(t7.instructions[1].translated &&
                    t7.instructions[1].opcode == "llvm.bitreverse.i64",
                "brev.b64 → llvm.bitreverse.i64"))
        return 1;

    // ── Test 8: warpsize / laneid / lanemask special registers ───────────────
    const std::string ptx8 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k8(.param .u64 p0)
{
    mov.u32 %r0, %warpsize;
    mov.u32 %r1, %laneid;
    mov.u32 %r2, %lanemask_eq;
    mov.u32 %r3, %lanemask_lt;
    mov.u32 %r4, %lanemask_gt;
    ret;
})PTX";

    const auto parsed8 = cumetal::ptx::parse_ptx(ptx8);
    if (!expect(parsed8.ok, "parse PTX8 warpsize/laneid")) return 1;

    const auto t8 = cumetal::passes::lower_intrinsics(parsed8.module.entries[0]);
    if (!expect(t8.ok, "lower PTX8 ok")) return 1;
    if (!expect(t8.warnings.empty(), "no warnings PTX8")) return 1;

    if (!expect(t8.instructions[0].translated &&
                    t8.instructions[0].operands.size() == 2 &&
                    t8.instructions[0].operands[1] == "air.constant.warp_size",
                "%warpsize → air.constant.warp_size"))
        return 1;
    if (!expect(t8.instructions[1].translated &&
                    t8.instructions[1].operands[1] == "air.thread_position_in_simdgroup",
                "%laneid → air.thread_position_in_simdgroup"))
        return 1;
    if (!expect(t8.instructions[2].translated &&
                    t8.instructions[2].operands[1] == "air.simdgroup.lanemask_eq",
                "%lanemask_eq → air.simdgroup.lanemask_eq"))
        return 1;
    if (!expect(t8.instructions[3].translated &&
                    t8.instructions[3].operands[1] == "air.simdgroup.lanemask_lt",
                "%lanemask_lt → air.simdgroup.lanemask_lt"))
        return 1;
    if (!expect(t8.instructions[4].translated &&
                    t8.instructions[4].operands[1] == "air.simdgroup.lanemask_gt",
                "%lanemask_gt → air.simdgroup.lanemask_gt"))
        return 1;

    // ── Test 9: f32/f64 math and 64-bit bitwise ops ──────────────────────────
    const std::string ptx9 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k9(.param .u64 p0, .param .u64 p1)
{
    add.f32 %f1, %f2, %f3;
    sub.f32 %f1, %f2, %f3;
    mul.f32 %f1, %f2, %f3;
    neg.f64 %fd1, %fd2;
    abs.f64 %fd1, %fd2;
    min.f64 %fd1, %fd2, %fd3;
    max.f64 %fd1, %fd2, %fd3;
    and.b64 %rd1, %rd2, %rd3;
    or.b64  %rd1, %rd2, %rd3;
    xor.b64 %rd1, %rd2, %rd3;
    not.b64 %rd1, %rd2;
    ret;
})PTX";

    const auto parsed9 = cumetal::ptx::parse_ptx(ptx9);
    if (!expect(parsed9.ok, "parse PTX9")) return 1;
    if (!expect(parsed9.module.entries.size() == 1, "single PTX9 entry")) return 1;

    const auto t9 = cumetal::passes::lower_intrinsics(parsed9.module.entries[0]);
    if (!expect(t9.ok, "lower PTX9 ok")) return 1;
    if (!expect(t9.warnings.empty(), "no warnings PTX9")) return 1;

    // f32 arithmetic
    if (!expect(t9.instructions[0].opcode == "llvm.add", "add.f32 → llvm.add")) return 1;
    if (!expect(t9.instructions[1].opcode == "llvm.sub", "sub.f32 → llvm.sub")) return 1;
    if (!expect(t9.instructions[2].opcode == "llvm.mul", "mul.f32 → llvm.mul")) return 1;
    // f64 unary/binary
    if (!expect(t9.instructions[3].opcode == "llvm.fneg", "neg.f64 → llvm.fneg")) return 1;
    if (!expect(t9.instructions[4].opcode == "llvm.fabs", "abs.f64 → llvm.fabs")) return 1;
    if (!expect(t9.instructions[5].opcode == "llvm.fmin", "min.f64 → llvm.fmin")) return 1;
    if (!expect(t9.instructions[6].opcode == "llvm.fmax", "max.f64 → llvm.fmax")) return 1;
    // 64-bit bitwise ops (root: and/or/xor/not — same lowering as 32-bit variants)
    if (!expect(t9.instructions[7].opcode == "llvm.and", "and.b64 → llvm.and")) return 1;
    if (!expect(t9.instructions[8].opcode == "llvm.or",  "or.b64  → llvm.or"))  return 1;
    if (!expect(t9.instructions[9].opcode == "llvm.xor", "xor.b64 → llvm.xor")) return 1;
    if (!expect(t9.instructions[10].opcode == "llvm.not","not.b64 → llvm.not")) return 1;

    // ── Test 10: isspacep / bfe / bfi / prmt ─────────────────────────────────
    const std::string ptx10 = R"PTX(
.version 8.0
.target sm_90
.visible .entry k10(.param .u64 p0)
{
    isspacep.global  %p1, %rd1;
    isspacep.shared  %p1, %rd1;
    bfe.u32 %r1, %r2, %r3, %r4;
    bfe.s32 %r1, %r2, %r3, %r4;
    bfi.b32 %r1, %r2, %r3, %r4, %r5;
    prmt.b32 %r1, %r2, %r3, %r4;
    ret;
})PTX";

    const auto parsed10 = cumetal::ptx::parse_ptx(ptx10);
    if (!expect(parsed10.ok, "parse PTX10 isspacep/bfe/bfi/prmt")) return 1;

    const auto t10 = cumetal::passes::lower_intrinsics(parsed10.module.entries[0]);
    if (!expect(t10.ok, "lower PTX10 ok")) return 1;
    if (!expect(t10.warnings.empty(), "no warnings PTX10")) return 1;

    if (!expect(t10.instructions[0].opcode == "air.isspacep.global",
                "isspacep.global → air.isspacep.global"))
        return 1;
    if (!expect(t10.instructions[1].opcode == "air.isspacep.nonglobal",
                "isspacep.shared → air.isspacep.nonglobal"))
        return 1;
    if (!expect(t10.instructions[2].opcode == "air.bfe.unsigned",
                "bfe.u32 → air.bfe.unsigned"))
        return 1;
    if (!expect(t10.instructions[3].opcode == "air.bfe.signed",
                "bfe.s32 → air.bfe.signed"))
        return 1;
    if (!expect(t10.instructions[4].opcode == "air.bfi",
                "bfi.b32 → air.bfi"))
        return 1;
    if (!expect(t10.instructions[5].opcode == "air.prmt",
                "prmt.b32 → air.prmt"))
        return 1;

    std::printf("PASS: intrinsic lower unit tests\n");
    return 0;
}
