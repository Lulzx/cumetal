#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-/usr/local}"

rm -f "$PREFIX/bin/air_inspect"
rm -f "$PREFIX/bin/air_validate"
rm -f "$PREFIX/bin/cumetal-air-emitter"

echo "Removed CuMetal phase-0.5 tooling from $PREFIX"
