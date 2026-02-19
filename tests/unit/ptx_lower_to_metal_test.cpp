#include "cumetal/ptx/lower_to_metal.h"

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
    // ── Test 1: skeleton PTX (no global memory) → matched=false ──────────────
    const std::string skeleton_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry negate(
    .param .u64 param_in,
    .param .u64 param_out
) {
    mov.u32 %r0, %tid.x;
    neg.f32 %f1, %f0;
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_negate;
    opts_negate.entry_name = "negate";
    const auto r_skeleton = cumetal::ptx::lower_ptx_to_metal_source(skeleton_ptx, opts_negate);
    if (!expect(r_skeleton.ok, "skeleton PTX lowering returns ok")) return 1;
    if (!expect(!r_skeleton.matched, "skeleton PTX (no global ops) not matched by generic emitter"))
        return 1;

    // ── Test 2: simple element-wise kernel via mad.lo.u32 GID ────────────────
    // clamp_relu: out[gid] = max(0.0f, in[gid])  (bounds-checked)
    const std::string relu_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry clamp_relu(
    .param .u64 clamp_relu_param_0,
    .param .u64 clamp_relu_param_1,
    .param .u32 clamp_relu_param_2
) {
    .reg .u64  %rd<4>;
    .reg .f32  %f<3>;
    .reg .u32  %r<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [clamp_relu_param_0];
    ld.param.u64 %rd1, [clamp_relu_param_1];
    ld.param.u32 %r0,  [clamp_relu_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;

    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra DONE;

    cvt.u64.u32 %rd2, %r4;
    shl.b64     %rd2, %rd2, 2;
    add.u64     %rd2, %rd0, %rd2;
    ld.global.f32 %f0, [%rd2];

    max.f32 %f1, %f0, 0.0;

    cvt.u64.u32 %rd3, %r4;
    shl.b64     %rd3, %rd3, 2;
    add.u64     %rd3, %rd1, %rd3;
    st.global.f32 [%rd3], %f1;

DONE:
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_relu;
    opts_relu.entry_name = "clamp_relu";
    const auto r_relu = cumetal::ptx::lower_ptx_to_metal_source(relu_ptx, opts_relu);
    if (!expect(r_relu.ok, "clamp_relu lowering ok")) return 1;
    if (!expect(r_relu.matched, "clamp_relu matched by generic emitter")) return 1;
    if (!expect(contains(r_relu.metal_source, "kernel void clamp_relu("),
                "clamp_relu kernel signature emitted"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "device float* clamp_relu_param_0"),
                "clamp_relu input pointer mapped"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "device float* clamp_relu_param_1"),
                "clamp_relu output pointer mapped"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "constant uint& clamp_relu_param_2"),
                "clamp_relu scalar count mapped as constant ref"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "uint gid [[thread_position_in_grid]]"),
                "clamp_relu thread position arg present"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "if (gid >= (uint)clamp_relu_param_2) return;"),
                "clamp_relu bounds guard emitted"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "clamp_relu_param_0[gid]"),
                "clamp_relu global load from param_0"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "clamp_relu_param_1[gid]"),
                "clamp_relu global store to param_1"))
        return 1;

    // ── Test 3: two-instruction GID (mul.lo.u32 + add.u32 pattern) ───────────
    // axpy_twoinstr: out[gid] = a * in[gid] + b
    const std::string twogid_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry axpy_twogid(
    .param .u64 axpy_twogid_param_0,
    .param .u64 axpy_twogid_param_1,
    .param .u32 axpy_twogid_param_2
) {
    .reg .u64  %rd<4>;
    .reg .f32  %f<4>;
    .reg .u32  %r<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [axpy_twogid_param_0];
    ld.param.u64 %rd1, [axpy_twogid_param_1];
    ld.param.u32 %r0,  [axpy_twogid_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    mov.u32 %r4, %tid.x;
    add.u32 %r5, %r3, %r4;

    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra DONE;

    cvt.u64.u32 %rd2, %r5;
    shl.b64     %rd2, %rd2, 2;
    add.u64     %rd2, %rd0, %rd2;
    ld.global.f32 %f0, [%rd2];

    fma.rn.f32 %f2, %f0, 2.0, 1.0;

    cvt.u64.u32 %rd3, %r5;
    shl.b64     %rd3, %rd3, 2;
    add.u64     %rd3, %rd1, %rd3;
    st.global.f32 [%rd3], %f2;

DONE:
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_twogid;
    opts_twogid.entry_name = "axpy_twogid";
    const auto r_twogid = cumetal::ptx::lower_ptx_to_metal_source(twogid_ptx, opts_twogid);
    if (!expect(r_twogid.ok, "axpy_twogid lowering ok")) return 1;
    if (!expect(r_twogid.matched,
                "axpy_twogid matched by generic emitter (mul.lo.u32+add.u32 GID pattern)"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "kernel void axpy_twogid("),
                "axpy_twogid kernel signature emitted"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "if (gid >= (uint)axpy_twogid_param_2) return;"),
                "axpy_twogid bounds guard emitted"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "axpy_twogid_param_0[gid]"),
                "axpy_twogid global load from param_0"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "axpy_twogid_param_1[gid]"),
                "axpy_twogid global store to param_1"))
        return 1;

    // ── Test 4: unsupported instruction → matched=false (no crash) ───────────
    const std::string unsup_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry unsup_generic(
    .param .u64 unsup_generic_param_0,
    .param .u64 unsup_generic_param_1,
    .param .u32 unsup_generic_param_2
) {
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.global.f32 %f0, [%rd0];
    foo.bar.unknown %f1, %f0;
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_unsup;
    opts_unsup.entry_name = "unsup_generic";
    const auto r_unsup = cumetal::ptx::lower_ptx_to_metal_source(unsup_ptx, opts_unsup);
    if (!expect(r_unsup.ok, "unsupported instruction lowering returns ok=true")) return 1;
    if (!expect(!r_unsup.matched, "unsupported instruction not matched by generic emitter")) return 1;

    std::printf("PASS: ptx lower-to-metal unit tests\n");
    return 0;
}
