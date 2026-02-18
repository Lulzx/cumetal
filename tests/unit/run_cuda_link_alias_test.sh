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
SRC="$WORK_DIR/link_cuda_alias_smoke.cpp"
BIN="$WORK_DIR/link_cuda_alias_smoke"

cat >"$SRC" <<'EOF'
#include "cuda.h"

#include <cstdio>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    int version = 0;
    if (cuDriverGetVersion(&version) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDriverGetVersion failed\n");
        return 1;
    }
    if (version <= 0) {
        std::fprintf(stderr, "FAIL: invalid driver version %d\n", version);
        return 1;
    }

    std::printf("PASS: linked via -lcuda alias\n");
    return 0;
}
EOF

"$CXX_COMPILER" \
  -std=c++20 \
  -I"$INCLUDE_DIR" \
  "$SRC" \
  -L"$LIB_DIR" \
  -Wl,-rpath,"$LIB_DIR" \
  -lcuda \
  -o "$BIN"

"$BIN"
