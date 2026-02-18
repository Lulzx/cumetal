#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLMC_DIR="${CUMETAL_LLMC_DIR:-${1:-}}"
if [[ -z "${LLMC_DIR}" ]]; then
    echo "usage: $0 <llm.c-dir> (or set CUMETAL_LLMC_DIR)" >&2
    exit 2
fi
if [[ ! -d "${LLMC_DIR}" ]]; then
    echo "llm.c directory not found: ${LLMC_DIR}" >&2
    exit 2
fi

CLANG_BIN="${CUMETAL_LLMC_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
if [[ ! -x "${CLANG_BIN}" ]]; then
    CLANG_BIN="$(command -v clang++ || true)"
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "clang++ not found" >&2
    exit 2
fi

CUDA_ARCH="${CUMETAL_LLMC_CUDA_ARCH:-sm_80}"
OUTPUT_NAME="${CUMETAL_LLMC_TEST_BINARY:-test_gpt2fp32cu}"
OBJ_DIR="${LLMC_DIR}/build/cumetal"
OBJ_FILE="${OBJ_DIR}/test_gpt2_fp32.cumetal.o"
OUT_FILE="${LLMC_DIR}/${OUTPUT_NAME}"

mkdir -p "${OBJ_DIR}"

PATH="${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
"${CLANG_BIN}" \
    -x cuda \
    -std=c++17 \
    -O2 \
    -DNDEBUG \
    --cuda-gpu-arch="${CUDA_ARCH}" \
    -nocudainc \
    -nocudalib \
    -I"${ROOT_DIR}/runtime/api" \
    -I"${LLMC_DIR}" \
    -c "${LLMC_DIR}/test_gpt2_fp32.cu" \
    -o "${OBJ_FILE}"

xcrun clang++ \
    "${OBJ_FILE}" \
    -L"${ROOT_DIR}/build" \
    -lcumetal \
    -Wl,-rpath,"${ROOT_DIR}/build" \
    -o "${OUT_FILE}"

echo "built ${OUT_FILE}"
