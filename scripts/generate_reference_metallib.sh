#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REF_DIR="$ROOT_DIR/tests/air_abi/reference"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"

METAL_FILE="$REF_DIR/vector_add.metal"
AIR_FILE="$REF_DIR/reference.air"
METALLIB_FILE="$REF_DIR/reference.metallib"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "xcrun not found" >&2
  exit 1
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "xcrun metal is unavailable. Install full Xcode to generate reference metallib." >&2
  exit 1
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "xcrun metallib is unavailable. Install full Xcode to generate reference metallib." >&2
  exit 1
fi

xcrun metal -c "$METAL_FILE" -o "$AIR_FILE"
xcrun metallib "$AIR_FILE" -o "$METALLIB_FILE"

echo "Generated: $METALLIB_FILE"

if [[ -x "$BUILD_DIR/air_inspect" ]]; then
  "$BUILD_DIR/air_inspect" "$METALLIB_FILE"
fi

if [[ -x "$BUILD_DIR/air_validate" ]]; then
  "$BUILD_DIR/air_validate" "$METALLIB_FILE" --xcrun || true
fi
