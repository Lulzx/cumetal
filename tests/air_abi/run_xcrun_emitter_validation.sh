#!/usr/bin/env bash
set -euo pipefail

EMITTER="$1"
VALIDATOR="$2"
INPUT_METAL="$3"
OUTPUT_METALLIB="$4"
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

echo "PASS: xcrun emitter validation completed"
