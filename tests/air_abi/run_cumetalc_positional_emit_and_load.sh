#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
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

"$CUMETALC" \
  --mode xcrun \
  "$INPUT_METAL" \
  -o "$OUTPUT_METALLIB" \
  --overwrite

"$LOAD_TEST" "$OUTPUT_METALLIB"

echo "PASS: cumetalc positional CLI output loads via MTLDevice.newLibraryWithData"
