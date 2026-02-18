#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
EMITTER="$2"
VALIDATOR="$3"
INPUT_PTX="$4"
OUTPUT_LL="$5"
OUTPUT_METALLIB="$6"

"$PTX2LLVM" \
  --input "$INPUT_PTX" \
  --output "$OUTPUT_LL" \
  --entry matrix_mul \
  --strict \
  --overwrite

rg "define void @matrix_mul" "$OUTPUT_LL" >/dev/null
rg "fmul float" "$OUTPUT_LL" >/dev/null
rg "fadd float" "$OUTPUT_LL" >/dev/null

"$EMITTER" \
  --mode experimental \
  --input "$OUTPUT_LL" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --require-function-list \
  --require-metadata

echo "PASS: matrix PTX->LLVM->experimental metallib validation completed"
