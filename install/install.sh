#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
PREFIX="${2:-/opt/cumetal}"
SHELL_RC="${CUMETAL_SHELL_RC:-${HOME}/.zshrc}"

MARKER_BEGIN="# >>> cumetal >>>"
MARKER_END="# <<< cumetal <<<"

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

cat >> "$SHELL_RC" <<EOF
$MARKER_BEGIN
export PATH="$PREFIX/bin:\$PATH"
export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}"
$MARKER_END
EOF

echo "Installed CuMetal to $PREFIX"
echo "Updated $SHELL_RC with CuMetal environment settings"
