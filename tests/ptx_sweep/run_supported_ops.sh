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

# Warp / SIMD-group primitives
run_case "sweep_shfl_sync_idx"   "shfl.sync.idx.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_down"  "shfl.sync.down.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_up"    "shfl.sync.up.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_bfly"  "shfl.sync.bfly.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_vote_sync_ballot" "vote.sync.ballot.b32 %r1, %p1, 0xffffffff;"
run_case "sweep_vote_sync_any"   "vote.sync.any.pred %p1, %p2, 0xffffffff;"
run_case "sweep_vote_sync_all"   "vote.sync.all.pred %p1, %p2, 0xffffffff;"
run_case "sweep_bar_warp_sync"   "bar.warp.sync 0xffffffff;"

# Memory barriers
run_case "sweep_membar_gl"   "membar.gl;"
run_case "sweep_membar_cta"  "membar.cta;"
run_case "sweep_membar_sys"  "membar.sys;"

# Async copy
run_case "sweep_cp_async_ca"       "cp.async.ca.shared.global [%rd1], [%rd2], 16;"
run_case "sweep_cp_async_commit"   "cp.async.commit_group;"
run_case "sweep_cp_async_wait_all" "cp.async.wait_all;"

# Warp reductions
run_case "sweep_redux_sync_add_s32" "redux.sync.add.s32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_add_f32" "redux.sync.add.f32 %f1, %f2, 0xffffffff;"
run_case "sweep_redux_sync_and_b32" "redux.sync.and.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_or_b32"  "redux.sync.or.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_xor_b32" "redux.sync.xor.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_min_s32" "redux.sync.min.s32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_max_s32" "redux.sync.max.s32 %r1, %r2, 0xffffffff;"

# Math intrinsics (extended)
run_case "sweep_sqrt_rn_f32"    "sqrt.rn.f32 %f1, %f2;"
run_case "sweep_rsqrt_approx"   "rsqrt.approx.f32 %f1, %f2;"
run_case "sweep_ex2_approx_f32" "ex2.approx.f32 %f1, %f2;"
run_case "sweep_lg2_approx_f32" "lg2.approx.f32 %f1, %f2;"
run_case "sweep_sin_approx_f32" "sin.approx.f32 %f1, %f2;"
run_case "sweep_cos_approx_f32" "cos.approx.f32 %f1, %f2;"
run_case "sweep_fma_rn_f32"     "fma.rn.f32 %f1, %f2, %f3, %f4;"
run_case "sweep_abs_f32"        "abs.f32 %f1, %f2;"
run_case "sweep_min_f32"        "min.f32 %f1, %f2, %f3;"
run_case "sweep_max_f32"        "max.f32 %f1, %f2, %f3;"

echo "PASS: PTX sweep supported-op strict checks completed"
