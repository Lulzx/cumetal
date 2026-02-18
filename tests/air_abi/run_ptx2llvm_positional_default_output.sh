#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
INPUT_PTX="$2"
WORKDIR="$3"

TMP_DIR="$WORKDIR/ptx2llvm-positional-default"
mkdir -p "$TMP_DIR"

PTX_COPY="$TMP_DIR/vector_add.ptx"
cp "$INPUT_PTX" "$PTX_COPY"

"$PTX2LLVM" \
  "$PTX_COPY" \
  --entry vector_add \
  --strict \
  --overwrite

OUT_LL="${PTX_COPY%.ptx}.ll"
if [[ ! -f "$OUT_LL" ]]; then
  echo "FAIL: expected default LLVM output at $OUT_LL"
  exit 1
fi

rg "define void @vector_add" "$OUT_LL" >/dev/null
rg "fadd float" "$OUT_LL" >/dev/null

echo "PASS: ptx2llvm positional/default-output flow completed"
