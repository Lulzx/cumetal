#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
VALIDATOR="$2"
INSPECTOR="$3"
INPUT_PTX="$4"
OUTPUT_METALLIB="$5"

"$CUMETALC" \
  --mode experimental \
  --input "$INPUT_PTX" \
  --output "$OUTPUT_METALLIB" \
  --entry vector_add \
  --ptx-strict \
  --overwrite

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$("$INSPECTOR" "$OUTPUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 1' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "vector_add"' >/dev/null

echo "PASS: cumetalc PTX experimental emit+validate completed"
