#!/usr/bin/env bash
set -euo pipefail

RUNTIME_LIB="$1"

if [[ ! -f "$RUNTIME_LIB" ]]; then
  echo "FAIL: runtime library missing at $RUNTIME_LIB"
  exit 1
fi

if ! command -v nm >/dev/null 2>&1; then
  echo "SKIP: nm not available"
  exit 77
fi

EXPORTED_SYMBOLS="$(nm -gU "$RUNTIME_LIB" | awk '{print $3}' | sed 's/^_//')"

assert_symbol() {
  local symbol="$1"
  if ! grep -Fxq "$symbol" <<<"$EXPORTED_SYMBOLS"; then
    echo "FAIL: expected exported symbol '$symbol' not found"
    exit 1
  fi
}

assert_symbol "__cudaRegisterFatBinary"
assert_symbol "__cudaRegisterFatBinary2"
assert_symbol "__cudaRegisterFatBinary3"
assert_symbol "__cudaRegisterFatBinaryEnd"
assert_symbol "__cudaUnregisterFatBinary"
assert_symbol "__cudaRegisterFunction"
assert_symbol "__cudaRegisterVar"
assert_symbol "__cudaRegisterManagedVar"
assert_symbol "__cudaPushCallConfiguration"
assert_symbol "__cudaPopCallConfiguration"
assert_symbol "cudaConfigureCall"
assert_symbol "cudaSetupArgument"
assert_symbol "cudaLaunch"

echo "PASS: runtime exports binary-shim compatibility symbols"
