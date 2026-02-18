#!/usr/bin/env bash
set -euo pipefail

RUNTIME_LIB="$1"
CBLAS_ALIAS="$2"
CURAND_ALIAS="$3"

if [[ ! -f "$RUNTIME_LIB" ]]; then
  echo "FAIL: runtime library missing at $RUNTIME_LIB"
  exit 1
fi

if [[ ! -L "$CBLAS_ALIAS" ]]; then
  echo "FAIL: expected libcublas alias symlink at $CBLAS_ALIAS"
  exit 1
fi

if [[ ! -L "$CURAND_ALIAS" ]]; then
  echo "FAIL: expected libcurand alias symlink at $CURAND_ALIAS"
  exit 1
fi

CBLAS_TARGET="$(readlink "$CBLAS_ALIAS")"
CURAND_TARGET="$(readlink "$CURAND_ALIAS")"

if [[ "$CBLAS_TARGET" != "$(basename "$RUNTIME_LIB")" ]]; then
  echo "FAIL: libcublas alias target mismatch ($CBLAS_TARGET)"
  exit 1
fi

if [[ "$CURAND_TARGET" != "$(basename "$RUNTIME_LIB")" ]]; then
  echo "FAIL: libcurand alias target mismatch ($CURAND_TARGET)"
  exit 1
fi

echo "PASS: libcublas/libcurand aliases point to $(basename "$RUNTIME_LIB")"
