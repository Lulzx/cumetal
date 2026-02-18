#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
LOAD_TEST="$2"
INPUT_PTX="$3"
OUTPUT_METALLIB="$4"
ENTRY_NAME="${5:-vector_add}"

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

if ! command -v llvm-as >/dev/null 2>&1; then
  echo "SKIP: llvm-as not available for PTX->LLVM->xcrun flow"
  exit 77
fi

"$CUMETALC" \
  --mode xcrun \
  --input "$INPUT_PTX" \
  --output "$OUTPUT_METALLIB" \
  --entry "$ENTRY_NAME" \
  --ptx-strict \
  --overwrite

"$LOAD_TEST" "$OUTPUT_METALLIB"

echo "PASS: cumetalc PTX xcrun output loads via MTLDevice.newLibraryWithData ($ENTRY_NAME)"
