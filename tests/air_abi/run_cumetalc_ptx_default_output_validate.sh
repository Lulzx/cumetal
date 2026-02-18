#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
VALIDATOR="$2"
INSPECTOR="$3"
INPUT_PTX="$4"
WORKDIR="$5"

TMP_DIR="$WORKDIR/cumetalc-ptx-default-output"
mkdir -p "$TMP_DIR"

PTX_COPY="$TMP_DIR/vector_add.ptx"
cp "$INPUT_PTX" "$PTX_COPY"

"$CUMETALC" \
  --mode experimental \
  "$PTX_COPY" \
  --entry vector_add \
  --ptx-strict \
  --overwrite

OUT_METALLIB="${PTX_COPY%.ptx}.metallib"
if [[ ! -f "$OUT_METALLIB" ]]; then
  echo "FAIL: expected default output metallib at $OUT_METALLIB"
  exit 1
fi

"$VALIDATOR" \
  "$OUT_METALLIB" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$("$INSPECTOR" "$OUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 1' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "vector_add"' >/dev/null

echo "PASS: cumetalc PTX default output validation completed"
