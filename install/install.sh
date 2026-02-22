#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
PREFIX="${2:-/opt/cumetal}"

MARKER_BEGIN="# >>> cumetal >>>"
MARKER_END="# <<< cumetal <<<"

# Detect shell and pick the right config file + syntax.
# CUMETAL_SHELL_RC overrides auto-detection.
if [[ -n "${CUMETAL_SHELL_RC:-}" ]]; then
  SHELL_RC="$CUMETAL_SHELL_RC"
  IS_FISH=0
  if [[ "$SHELL_RC" == *config.fish ]]; then IS_FISH=1; fi
elif [[ "${SHELL:-}" == */fish ]]; then
  SHELL_RC="${HOME}/.config/fish/config.fish"
  IS_FISH=1
else
  SHELL_RC="${HOME}/.zshrc"
  IS_FISH=0
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "build directory not found: $BUILD_DIR" >&2
  exit 1
fi

cmake --install "$BUILD_DIR" --prefix "$PREFIX"

install -m 755 "$(dirname "$0")/uninstall.sh" "$PREFIX/uninstall.sh"

mkdir -p "$(dirname "$SHELL_RC")"
touch "$SHELL_RC"

if grep -qF "$MARKER_BEGIN" "$SHELL_RC"; then
  tmp="$(mktemp)"
  awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    skip != 1 {print}
  ' "$SHELL_RC" > "$tmp"
  mv "$tmp" "$SHELL_RC"
fi

if [[ "$IS_FISH" -eq 1 ]]; then
  cat >> "$SHELL_RC" <<EOF
$MARKER_BEGIN
set -gx PATH "$PREFIX/bin" \$PATH
set -gx DYLD_FALLBACK_LIBRARY_PATH "$PREFIX/lib" \$DYLD_FALLBACK_LIBRARY_PATH
$MARKER_END
EOF
else
  cat >> "$SHELL_RC" <<EOF
$MARKER_BEGIN
export PATH="$PREFIX/bin:\$PATH"
export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}"
$MARKER_END
EOF
fi

echo "Installed CuMetal to $PREFIX"
echo "Updated $SHELL_RC with CuMetal environment settings"
