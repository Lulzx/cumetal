#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
WORKDIR="$2"
OUT_DIR="$WORKDIR/ptx_sweep_supported"
mkdir -p "$OUT_DIR"

run_case() {
  local case_name="$1"
  local instruction="$2"
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
  ${instruction}
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
run_case "sweep_mul_lo_s32" "mul.lo.s32 %r1, %r2, %r3;"
run_case "sweep_mad_lo_s32" "mad.lo.s32 %r1, %r2, %r3, %r4;"
run_case "sweep_mov_tid_x" "mov.u32 %r1, %tid.x;"
run_case "sweep_ld_shared_u32" "ld.shared.u32 %r1, [%rd1];"
run_case "sweep_st_shared_u32" "st.shared.u32 [%rd1], %r1;"
run_case "sweep_bar_sync" "bar.sync 0;"

echo "PASS: PTX sweep supported-op strict checks completed"
