#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-/opt/cumetal}"
SHELL_RC="${CUMETAL_SHELL_RC:-${HOME}/.zshrc}"

MARKER_BEGIN="# >>> cumetal >>>"
MARKER_END="# <<< cumetal <<<"

rm -f "$PREFIX/bin/air_inspect"
rm -f "$PREFIX/bin/air_validate"
rm -f "$PREFIX/bin/cumetal-air-emitter"
rm -f "$PREFIX/bin/cumetalc"
rm -f "$PREFIX/lib/libcumetal.dylib"
rm -f "$PREFIX/lib/libcublas.dylib"
rm -f "$PREFIX/lib/libcufft.dylib"
rm -f "$PREFIX/lib/libcurand.dylib"
rm -f "$PREFIX/include/cuda.h"
rm -f "$PREFIX/include/cuda_runtime.h"
rm -f "$PREFIX/include/cufft.h"
rm -f "$PREFIX/include/cublas_v2.h"
rm -f "$PREFIX/include/curand.h"
rm -f "$PREFIX/uninstall.sh"

if [[ -f "$SHELL_RC" ]] && grep -qF "$MARKER_BEGIN" "$SHELL_RC"; then
  tmp="$(mktemp)"
  awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    skip != 1 {print}
  ' "$SHELL_RC" > "$tmp"
  mv "$tmp" "$SHELL_RC"
fi

rmdir "$PREFIX/bin" 2>/dev/null || true
rmdir "$PREFIX/lib" 2>/dev/null || true
rmdir "$PREFIX/include" 2>/dev/null || true
rmdir "$PREFIX" 2>/dev/null || true

echo "Removed CuMetal from $PREFIX"
echo "Removed CuMetal environment settings from $SHELL_RC"
