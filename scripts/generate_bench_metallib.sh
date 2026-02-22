#!/usr/bin/env bash
# generate_bench_metallib.sh — compile bench_kernels.metal to bench_kernels.metallib
# Usage: generate_bench_metallib.sh [OUTPUT_DIR]
# If xcrun metal/metallib are unavailable the script exits with code 77 (CTest skip).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/tools/cumetal_bench"
OUTPUT_DIR="${1:-$BENCH_DIR}"

METAL_FILE="$BENCH_DIR/bench_kernels.metal"
AIR_FILE="$OUTPUT_DIR/bench_kernels.air"
METALLIB_FILE="$OUTPUT_DIR/bench_kernels.metallib"

if ! command -v xcrun >/dev/null 2>&1; then
    echo "xcrun not found — skipping bench metallib generation" >&2
    exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "xcrun metal not available — skipping bench metallib generation" >&2
    exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
    echo "xcrun metallib not available — skipping bench metallib generation" >&2
    exit 77
fi

mkdir -p "$OUTPUT_DIR"

xcrun metal -c "$METAL_FILE" -o "$AIR_FILE"
xcrun metallib "$AIR_FILE" -o "$METALLIB_FILE"

echo "Generated: $METALLIB_FILE"
