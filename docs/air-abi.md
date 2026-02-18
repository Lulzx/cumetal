# AIR / metallib ABI Notes (Phase 0.5)

Status: draft, continuously updated as reverse-engineering data is collected.

## Scope

This document records container-level findings for `.metallib` validation and CuMetal's
Phase 0.5 harness. The focus is structural acceptance, not kernel execution.

## Tooling in this repo

- `air_inspect`: fast structural dump (`magic`, header words, embedded LLVM bitcode signatures,
  string candidates).
- `cumetal-air-emitter`: emits a `.metallib` using one of:
  - `xcrun` mode: packages `.air` payloads with Apple tools (or `.metal` via `xcrun metal`).
  - `experimental` mode: CuMetal-owned provisional container format for local iteration.
- `air_validate`: validates container shape and bitcode signatures, optional `xcrun metal -validate`.
- `cumetal_metal_load_test`: checks `MTLDevice.newLibraryWithData:` acceptance.

## Current findings

- A production metallib should have a magic prefix that begins with `MTL`.
- Embedded LLVM bitcode signatures are discoverable and can be used for sanity checks during
  development.
- Validation should be treated as a pipeline, not a single check:
  1. local parser (`air_validate`)
  2. Apple CLI validation (`xcrun metal -validate`) when available
  3. runtime load validation (`newLibraryWithData:`)

## Repro workflow

```bash
# 1) Build tools
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# 2) Generate reference assets (requires full Xcode shader tools)
./scripts/generate_reference_metallib.sh

# 3) Inspect + validate
./build/air_inspect tests/air_abi/reference/reference.metallib
./build/air_validate tests/air_abi/reference/reference.metallib --xcrun

# 4) Runtime acceptance
ctest --test-dir build -R air_abi_metal_load --output-on-failure
```

## Environment caveat

On systems where `xcrun metal` / `xcrun metallib` are unavailable (for example, Command Line Tools
without full Xcode shader utilities), use `cumetal-air-emitter --mode experimental` for local
container development and parser tests. Driver acceptance checks will be skipped/fail in that mode.

## Next updates expected

- Replace provisional header assumptions with byte-level field mapping from confirmed reference
  metallibs across Xcode versions.
- Add per-function metadata extraction in `air_inspect`.
- Gate CI with known-good reference metallibs in `tests/air_abi/`.
