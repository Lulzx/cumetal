#!/usr/bin/env bash
set -euo pipefail

AIR_INSPECT="$1"
SOURCE_METAL="$2"
WORKDIR="$3"

XCODE15_DIR="${CUMETAL_XCODE15_DEVELOPER_DIR:-}"
XCODE16_DIR="${CUMETAL_XCODE16_DEVELOPER_DIR:-}"

DEFAULT_DEVELOPER_DIR="$(xcode-select -p 2>/dev/null || true)"
if [[ -z "$XCODE15_DIR" ]]; then
  XCODE15_DIR="$DEFAULT_DEVELOPER_DIR"
fi
if [[ -z "$XCODE16_DIR" ]]; then
  XCODE16_DIR="$DEFAULT_DEVELOPER_DIR"
fi

if [[ -z "$XCODE15_DIR" || -z "$XCODE16_DIR" ]]; then
  echo "SKIP: no Xcode developer directory found"
  echo "      set CUMETAL_XCODE15_DEVELOPER_DIR/CUMETAL_XCODE16_DEVELOPER_DIR or run xcode-select --switch"
  exit 77
fi

if [[ ! -d "$XCODE15_DIR" || ! -d "$XCODE16_DIR" ]]; then
  echo "SKIP: configured Xcode developer directories do not exist"
  exit 77
fi

if [[ "$XCODE15_DIR" == "$XCODE16_DIR" ]]; then
  echo "INFO: single-Xcode fallback mode (matrix compare uses the same developer dir):"
  echo "      $XCODE15_DIR"
fi

mkdir -p "$WORKDIR"

run_for_xcode() {
  local label="$1"
  local developer_dir="$2"
  local out_txt="$3"
  local out_air="$WORKDIR/${label}.air"
  local out_metallib="$WORKDIR/${label}.metallib"

  if ! DEVELOPER_DIR="$developer_dir" xcrun --find metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal not available for ${label} at ${developer_dir}"
    exit 77
  fi
  if ! DEVELOPER_DIR="$developer_dir" xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: xcrun metallib not available for ${label} at ${developer_dir}"
    exit 77
  fi

  DEVELOPER_DIR="$developer_dir" xcrun metal -c "$SOURCE_METAL" -o "$out_air"
  DEVELOPER_DIR="$developer_dir" xcrun metallib "$out_air" -o "$out_metallib"
  "$AIR_INSPECT" "$out_metallib" > "$out_txt"

  rg -q "^Magic: MTLB" "$out_txt" || { echo "FAIL: ${label} magic check failed"; exit 1; }
  rg -q "^Function count: 1$" "$out_txt" || { echo "FAIL: ${label} function-count check failed"; exit 1; }
  rg -q "\\[kernel 0\\] vector_add" "$out_txt" || { echo "FAIL: ${label} kernel-name check failed"; exit 1; }
  rg -q "air.version=2.8" "$out_txt" || { echo "FAIL: ${label} air.version check failed"; exit 1; }
  rg -q "language.version=4.0" "$out_txt" || { echo "FAIL: ${label} language.version check failed"; exit 1; }
}

OUT15="$WORKDIR/xcode15.inspect.txt"
OUT16="$WORKDIR/xcode16.inspect.txt"

run_for_xcode "xcode15" "$XCODE15_DIR" "$OUT15"
run_for_xcode "xcode16" "$XCODE16_DIR" "$OUT16"

HASH15="$(rg -o "function.hash.prefix=[0-9a-f]+" "$OUT15" | head -n1 || true)"
HASH16="$(rg -o "function.hash.prefix=[0-9a-f]+" "$OUT16" | head -n1 || true)"
if [[ -z "$HASH15" || -z "$HASH16" ]]; then
  echo "FAIL: failed to extract hash prefixes from one of the Xcode outputs"
  exit 1
fi

echo "PASS: Xcode matrix ABI regression checks succeeded"
echo "  xcode15: $HASH15"
echo "  xcode16: $HASH16"
