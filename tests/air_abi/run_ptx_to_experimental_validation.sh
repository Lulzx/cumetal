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
  --entry vector_add \
  --overwrite

"$EMITTER" \
  --mode experimental \
  --input "$OUTPUT_LL" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --require-function-list \
  --require-metadata

echo "PASS: PTX->LLVM->experimental metallib validation completed"
