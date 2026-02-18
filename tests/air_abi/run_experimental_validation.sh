#!/usr/bin/env bash
set -euo pipefail

EMITTER="$1"
VALIDATOR="$2"
INPUT_LL="$3"
OUT_METALLIB="$4"

"$EMITTER" \
  --mode experimental \
  --input "$INPUT_LL" \
  --output "$OUT_METALLIB" \
  --overwrite

"$VALIDATOR" \
  "$OUT_METALLIB" \
  --require-function-list \
  --require-metadata
