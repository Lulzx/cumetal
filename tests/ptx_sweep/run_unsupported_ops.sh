#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
WORKDIR="$2"
OUT_DIR="$WORKDIR/ptx_sweep_unsupported"
mkdir -p "$OUT_DIR"

PTX_FILE="$OUT_DIR/unsupported.ptx"
LL_FILE="$OUT_DIR/unsupported.ll"

cat >"$PTX_FILE" <<'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry unsupported(
  .param .u64 p0
)
{
  foo.bar %r1, %r2;
  ret;
}
EOF

if "$PTX2LLVM" --input "$PTX_FILE" --output "$LL_FILE" --entry unsupported --strict --overwrite; then
  echo "FAIL: strict PTX sweep expected unsupported opcode to fail"
  exit 1
fi

echo "PASS: PTX sweep unsupported-op strict rejection completed"
