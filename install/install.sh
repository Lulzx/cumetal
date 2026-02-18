#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
PREFIX="${2:-/usr/local}"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "build directory not found: $BUILD_DIR" >&2
  exit 1
fi

cmake --install "$BUILD_DIR" --prefix "$PREFIX"
echo "Installed CuMetal phase-0.5 tooling to $PREFIX"
