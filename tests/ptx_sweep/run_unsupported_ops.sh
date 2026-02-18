#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
WORKDIR="$2"
OUT_DIR="$WORKDIR/ptx_sweep_unsupported"
mkdir -p "$OUT_DIR"

run_case_expect_fail() {
  local case_name="$1"
  local instruction="$2"
  local ptx_file="$OUT_DIR/${case_name}.ptx"
  local ll_file="$OUT_DIR/${case_name}.ll"

  cat >"$ptx_file" <<EOF
.version 8.0
.target sm_90
.address_size 64
.visible .entry ${case_name}(
  .param .u64 p0
)
{
  ${instruction}
  ret;
}
EOF

  if "$PTX2LLVM" --input "$ptx_file" --output "$ll_file" --entry "$case_name" --strict --overwrite; then
    echo "FAIL: strict PTX sweep expected unsupported opcode to fail (${case_name})"
    exit 1
  fi
}

run_case_expect_fail "unsupported_foo" "foo.bar %r1, %r2;"
run_case_expect_fail "unsupported_trap" "trap;"
run_case_expect_fail "unsupported_tex" "tex.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [%rd1], %f5;"
run_case_expect_fail "unsupported_suld" "suld.1d.v4.u32.trap [%rd1], {%r1,%r2,%r3,%r4};"

echo "PASS: PTX sweep unsupported-op strict rejection completed"
