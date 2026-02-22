#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-/opt/cumetal}"

MARKER_BEGIN="# >>> cumetal >>>"
MARKER_END="# <<< cumetal <<<"

# Mirror the shell detection logic from install.sh.
if [[ -n "${CUMETAL_SHELL_RC:-}" ]]; then
  SHELL_RC="$CUMETAL_SHELL_RC"
elif [[ "${SHELL:-}" == */fish ]]; then
  SHELL_RC="${HOME}/.config/fish/config.fish"
else
  SHELL_RC="${HOME}/.zshrc"
fi

rm -f "$PREFIX/bin/air_inspect"
rm -f "$PREFIX/bin/air_validate"
rm -f "$PREFIX/bin/cumetal-air-emitter"
rm -f "$PREFIX/bin/cumetalc"
rm -f "$PREFIX/lib/libcumetal.dylib"
rm -f "$PREFIX/lib/libcublas.dylib"
rm -f "$PREFIX/lib/libcufft.dylib"
rm -f "$PREFIX/lib/libcurand.dylib"
rm -f "$PREFIX/include/cuda.h"
rm -f "$PREFIX/include/cuda_fp16.h"
rm -f "$PREFIX/include/cuda_runtime.h"
rm -f "$PREFIX/include/cufft.h"
rm -f "$PREFIX/include/cublas_v2.h"
rm -f "$PREFIX/include/curand.h"
rm -f "$PREFIX/include/cooperative_groups.h"
rm -f "$PREFIX/include/cooperative_groups/reduce.h"
rmdir "$PREFIX/include/cooperative_groups" 2>/dev/null || true
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
