#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <air-validate-binary> <work-dir>" >&2
  exit 2
fi

AIR_VALIDATE="$1"
WORK_DIR="$2"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

run_expect_failure() {
  local label="$1"
  shift
  local logfile="$WORK_DIR/${label}.log"

  if "$@" >"$logfile" 2>&1; then
    echo "FAIL: expected command to fail for $label" >&2
    cat "$logfile" >&2
    exit 1
  fi
}

# Missing file should fail with I/O error.
run_expect_failure missing "$AIR_VALIDATE" "$WORK_DIR/does_not_exist.metallib"

# Empty file should fail explicit empty-file check.
: > "$WORK_DIR/empty.metallib"
run_expect_failure empty "$AIR_VALIDATE" "$WORK_DIR/empty.metallib"

# Non-metallib payload should fail strict magic/bitcode validation.
printf 'NOTAMETALLIB\x00\x01\x02\x03' > "$WORK_DIR/bad_magic.metallib"
run_expect_failure bad_magic "$AIR_VALIDATE" "$WORK_DIR/bad_magic.metallib"

# Missing function list requirement should fail for malformed input.
run_expect_failure require_function_list "$AIR_VALIDATE" \
  "$WORK_DIR/bad_magic.metallib" --require-function-list

# Missing metadata requirement should fail for malformed input.
run_expect_failure require_metadata "$AIR_VALIDATE" \
  "$WORK_DIR/bad_magic.metallib" --require-metadata

echo "PASS: air_validate rejects malformed inputs as expected"
