#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
LOAD_TEST="$2"
INPUT_METAL="$3"
WORK_DIR="$4"

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

mkdir -p "$WORK_DIR"
WORK_INPUT="$WORK_DIR/vector_add.default_input.metal"
cp "$INPUT_METAL" "$WORK_INPUT"

"$CUMETALC" \
  --mode xcrun \
  "$WORK_INPUT" \
  --overwrite

EXPECTED_OUTPUT="${WORK_INPUT%.metal}.metallib"
if [[ ! -f "$EXPECTED_OUTPUT" ]]; then
  echo "FAIL: expected default output not found: $EXPECTED_OUTPUT" >&2
  exit 1
fi

"$LOAD_TEST" "$EXPECTED_OUTPUT"

echo "PASS: cumetalc default output path loads via MTLDevice.newLibraryWithData"
