#!/usr/bin/env bash
set -euo pipefail

CXX_COMPILER="$1"
INCLUDE_DIR="$2"
LIB_DIR="$3"
WORK_DIR="$4"

if [[ ! -x "$CXX_COMPILER" ]]; then
  echo "SKIP: C++ compiler not executable at $CXX_COMPILER"
  exit 77
fi

if [[ ! -d "$INCLUDE_DIR" || ! -d "$LIB_DIR" ]]; then
  echo "FAIL: include/lib directories missing"
  exit 1
fi

mkdir -p "$WORK_DIR"
SRC="$WORK_DIR/link_alias_smoke.cpp"
BIN="$WORK_DIR/link_alias_smoke"

cat >"$SRC" <<'EOF'
#include "cublas_v2.h"
#include "curand.h"

#include <cstdio>

int main() {
    cublasHandle_t cublas = nullptr;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasCreate failed\n");
        return 1;
    }

    curandGenerator_t curand = nullptr;
    if (curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandCreateGenerator failed\n");
        return 1;
    }

    if (curandDestroyGenerator(curand) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandDestroyGenerator failed\n");
        return 1;
    }
    if (cublasDestroy(cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDestroy failed\n");
        return 1;
    }

    std::printf("PASS: linked via -lcublas/-lcurand aliases\n");
    return 0;
}
EOF

"$CXX_COMPILER" \
  -std=c++20 \
  -I"$INCLUDE_DIR" \
  "$SRC" \
  -L"$LIB_DIR" \
  -Wl,-rpath,"$LIB_DIR" \
  -lcublas \
  -lcurand \
  -o "$BIN"

"$BIN"
