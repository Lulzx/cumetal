#!/usr/bin/env bash
set -euo pipefail

EMITTER="$1"
VALIDATOR="$2"
INSPECTOR="$3"
LOAD_TEST="$4"
INPUT_METAL="$5"
OUTPUT_METALLIB="$6"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MLA_CMD="swift run --package-path $ROOT_DIR/tools/metal_library_archive_bridge cumetal-mla-validate"

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

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --xcrun \
  --mla \
  --mla-cmd "$MLA_CMD" \
  --require-function-list \
  --require-metadata

"$LOAD_TEST" "$OUTPUT_METALLIB"

INSPECT_JSON="$("$INSPECTOR" "$OUTPUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 2' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "vector_add"' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "scale"' >/dev/null

echo "PASS: multi-kernel metallib emit/validate/load test completed"
