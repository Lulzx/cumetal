#!/usr/bin/env bash
set -euo pipefail

RUNTIME_LIB="$1"
CUDA_ALIAS="$2"

if [[ ! -f "$RUNTIME_LIB" ]]; then
  echo "FAIL: runtime library missing at $RUNTIME_LIB"
  exit 1
fi

if [[ ! -L "$CUDA_ALIAS" ]]; then
  echo "FAIL: expected libcuda alias symlink at $CUDA_ALIAS"
  exit 1
fi

CUDA_TARGET="$(readlink "$CUDA_ALIAS")"
if [[ "$CUDA_TARGET" != "$(basename "$RUNTIME_LIB")" ]]; then
  echo "FAIL: libcuda alias target mismatch ($CUDA_TARGET)"
  exit 1
fi

echo "PASS: libcuda alias points to $(basename "$RUNTIME_LIB")"
