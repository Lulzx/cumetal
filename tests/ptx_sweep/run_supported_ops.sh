#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
WORKDIR="$2"
OUT_DIR="$WORKDIR/ptx_sweep_supported"
mkdir -p "$OUT_DIR"

run_case() {
  local case_name="$1"
  local body="$2"
  local ptx_file="$OUT_DIR/${case_name}.ptx"
  local ll_file="$OUT_DIR/${case_name}.ll"

  cat >"$ptx_file" <<EOF
.version 8.0
.target sm_90
.address_size 64
.visible .entry ${case_name}(
  .param .u64 p0,
  .param .u64 p1
)
{
  ${body}
  ret;
}
EOF

  "$PTX2LLVM" \
    --input "$ptx_file" \
    --output "$ll_file" \
    --entry "$case_name" \
    --strict \
    --overwrite

  rg "define void @${case_name}" "$ll_file" >/dev/null
}

run_case "sweep_add_s32" "add.s32 %r1, %r2, %r3;"
run_case "sweep_sub_s32" "sub.s32 %r1, %r2, %r3;"
run_case "sweep_mul_lo_s32" "mul.lo.s32 %r1, %r2, %r3;"
run_case "sweep_div_s32" "div.s32 %r1, %r2, %r3;"
run_case "sweep_rem_s32" "rem.s32 %r1, %r2, %r3;"
run_case "sweep_mad_lo_s32" "mad.lo.s32 %r1, %r2, %r3, %r4;"
run_case "sweep_neg_f32" "neg.f32 %f1, %f2;"
run_case "sweep_shl_b64" "shl.b64 %rd1, %rd2, 2;"
run_case "sweep_not_b32" "not.b32 %r1, %r2;"
run_case "sweep_and_b32" "and.b32 %r1, %r2, %r3;"
run_case "sweep_or_b32" "or.b32 %r1, %r2, %r3;"
run_case "sweep_xor_b32" "xor.b32 %r1, %r2, %r3;"
run_case "sweep_selp_f32" "selp.f32 %f1, %f2, %f3, %p1;"
run_case "sweep_rcp_f32" "rcp.rn.f32 %f1, %f2;"
run_case "sweep_mov_tid_x" "mov.u32 %r1, %tid.x;"
run_case "sweep_mov_ctaid_y" "mov.u32 %r1, %ctaid.y;"
run_case "sweep_mov_ntid_z" "mov.u32 %r1, %ntid.z;"
run_case "sweep_mov_nctaid_x" "mov.u32 %r1, %nctaid.x;"
run_case "sweep_ld_shared_u32" "ld.shared.u32 %r1, [%rd1];"
run_case "sweep_st_shared_u32" "st.shared.u32 [%rd1], %r1;"
run_case "sweep_ld_global_f32" "ld.global.f32 %f1, [%rd1];"
run_case "sweep_st_global_f32" "st.global.f32 [%rd1], %f1;"
run_case "sweep_ld_local_u16" "ld.local.u16 %r1, [%rd1];"
run_case "sweep_st_local_u16" "st.local.u16 [%rd1], %r1;"
run_case "sweep_cvta_shared" "cvta.to.shared.u64 %rd2, %rd1;"
run_case "sweep_cvta_global" "cvta.to.global.u64 %rd2, %rd1;"
run_case "sweep_cvta_local" "cvta.to.local.u64 %rd2, %rd1;"
run_case "sweep_bar_sync" "bar.sync 0;"
run_case "sweep_setp_eq_s32" "setp.eq.s32 %p1, %r1, %r2;"
run_case "sweep_bra_label" $'bra L_done;\nL_done:'
run_case "sweep_call_vprintf" "call.uni (%r0), vprintf, (\"tid=%u\", %r1);"
run_case "sweep_atom_global_add_f32" "atom.global.add.f32 %f1, [%rd1], %f2;"

echo "PASS: PTX sweep supported-op strict checks completed"
