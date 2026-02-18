#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <build_dir> <bench_bin> <generate_reference_script> <reference_metallib>"
    exit 2
fi

BUILD_DIR="$1"
BENCH_BIN="$2"
GEN_SCRIPT="$3"
REFERENCE_METALLIB="$4"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "FAIL: build dir does not exist: $BUILD_DIR"
    exit 1
fi
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "FAIL: bench binary not executable: $BENCH_BIN"
    exit 1
fi
if [[ ! -x "$GEN_SCRIPT" ]]; then
    echo "FAIL: reference generation script not executable: $GEN_SCRIPT"
    exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not available"
    exit 77
fi

if [[ ! -f "$REFERENCE_METALLIB" ]]; then
    if ! "$GEN_SCRIPT" >/dev/null 2>&1; then
        echo "SKIP: unable to generate reference metallib (requires full Xcode toolchain)"
        exit 77
    fi
fi

PASS_LOG="$(mktemp)"
FAIL_LOG="$(mktemp)"
trap 'rm -f "$PASS_LOG" "$FAIL_LOG"' EXIT

if ! "$BENCH_BIN" \
        --metallib "$REFERENCE_METALLIB" \
        --kernel vector_add \
        --elements 32768 \
        --warmup 1 \
        --iterations 2 \
        --max-ratio 100 \
        >"$PASS_LOG" 2>&1; then
    cat "$PASS_LOG"
    echo "FAIL: benchmark should pass with relaxed ratio threshold"
    exit 1
fi

if "$BENCH_BIN" \
        --metallib "$REFERENCE_METALLIB" \
        --kernel vector_add \
        --elements 32768 \
        --warmup 1 \
        --iterations 2 \
        --max-ratio 1e-6 \
        >"$FAIL_LOG" 2>&1; then
    cat "$FAIL_LOG"
    echo "FAIL: benchmark should fail with an unrealistically strict ratio threshold"
    exit 1
fi

if ! grep -q "exceeds threshold" "$FAIL_LOG"; then
    cat "$FAIL_LOG"
    echo "FAIL: strict-threshold benchmark failure did not report ratio gate message"
    exit 1
fi

echo "PASS: benchmark ratio gate accepts relaxed threshold and rejects strict threshold"
