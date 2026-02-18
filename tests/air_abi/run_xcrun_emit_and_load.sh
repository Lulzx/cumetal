#!/usr/bin/env bash
set -euo pipefail

EMITTER="$1"
LOAD_TEST="$2"
INPUT_METAL="$3"
OUTPUT_METALLIB="$4"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal not available"
  exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: xcrun metallib not available"
  exit 77
fi

"$EMITTER" \
  --mode xcrun \
  --input "$INPUT_METAL" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$LOAD_TEST" "$OUTPUT_METALLIB"

echo "PASS: xcrun emitter output loads via MTLDevice.newLibraryWithData"
