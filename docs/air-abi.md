# AIR / metallib ABI Notes (Phase 0.5)

Status: draft with concrete reference bytes from:

- `tests/air_abi/reference/reference.metallib` (single kernel)
- `tests/air_abi/reference/multi_kernel.metal` compiled via `xcrun metal` + `xcrun metallib`

## Scope

This document records container-level findings for `.metallib` validation and CuMetal's
Phase 0.5 harness. The focus is structural acceptance, not kernel execution.

## Tooling in this repo

- `air_inspect`: structural dump with function-level output when available:
  - CuMetal experimental container v2 (`MTLB` + explicit function table)
  - Apple metallib function list tags (`NAME`, `MDSZ`, `OFFT`, `VERS`, `TYPE`, `HASH`) when parseable
- `cumetal-air-emitter`: emits a `.metallib` using one of:
  - `xcrun` mode: packages `.air` payloads with Apple tools (or `.metal` via `xcrun metal`).
  - `experimental` mode: CuMetal-owned provisional container format for local iteration.
- `air_validate`: validates container shape, function list, required metadata fields, and bitcode
  signatures. Optional checks:
  - `xcrun metal -validate`
  - `llvm-dis` bitcode verification (if installed)
  - MetalLibraryArchive bridge command (`tools/metal_library_archive_bridge`)
- `cumetal_metal_load_test`: checks `MTLDevice.newLibraryWithData:` acceptance.

## Reference layout snapshot

Reference artifact (`tests/air_abi/reference/reference.metallib`) from:

```bash
xcrun metal -c tests/air_abi/reference/vector_add.metal -o reference.air
xcrun metallib reference.air -o reference.metallib
```

`air_inspect` summary:

- Size: `3760` bytes (`0xeb0`)
- Magic: `MTLB`
- Function list parser: `metallib-function-list`
- Function count: `1` (`vector_add`)
- Kernel bitcode: offset `0xf0` size `0xdc0`

Header fields currently used by `parse_real_metallib` (`compiler/common/src/metallib.cpp`):

| File offset | Type | Meaning | Value (reference) |
|---|---|---|---|
| `0x18` | `u64` | function-list offset | `0x58` |
| `0x20` | `u64` | function-list size | `0x80` |
| `0x48` | `u64` | bitcode-section offset | `0xf0` |
| `0x50` | `u64` | bitcode-section size | `0xdc0` |

The parser accepts two size interpretations for function-list end:

1. `function_list_offset + function_list_size`
2. `function_list_offset + function_list_size + 4`

The `+4` variant is required for metallib layouts where the stored size excludes the leading
`entry_count` word.

## Function record tags

Within the function-list group, the following 4-byte tags are parsed today:

| Tag | Meaning | Reference value |
|---|---|---|
| `NAME` | function symbol | `vector_add` |
| `TYPE` | function kind | `2` (kernel) |
| `HASH` | digest blob (prefix reported) | `e64ad8cd3651085e...` |
| `MDSZ` | bitcode payload size | `0xdc0` |
| `OFFT` | public/private/bitcode offsets (relative to bitcode section) | `0/0/0` |
| `VERS` | AIR + language version (`u16` pairs) | AIR `2.8`, language `4.0` |
| `ENDT` | terminator | present (multiple) |

Observed bytes around the first record (`0x60`):

```text
4e 41 4d 45 ... 54 59 50 45 ... 48 41 53 48 ... 4d 44 53 5a ...
4f 46 46 54 ... 56 45 52 53 ... 45 4e 44 54
```

## Multi-kernel layout snapshot

Compiling `tests/air_abi/reference/multi_kernel.metal` with `xcrun` produces a two-function
metallib with the same tag schema:

- Size: `7307` bytes (`0x1c8b`)
- Function count: `2` (`vector_add`, `scale`)
- Bitcode sections:
  - `vector_add`: offset `0x17b`, size `0xdc0`
  - `scale`: offset `0xf3b`, size `0xd50`
- Metadata highlights:
  - both functions report `TYPE=2` (`kernel`)
  - both carry `VERS` AIR `2.8` and language `4.0`
  - second-function `OFFT` values are non-zero (`public=8`, `private=8`, `bitcode=3520`)

Practical parser notes from this sample:

1. Function-list entry order matches source declaration order.
2. `OFFT.bitcode` for later entries can point to byte ranges after prior kernels.
3. The core tag set (`NAME`, `TYPE`, `HASH`, `MDSZ`, `OFFT`, `VERS`, `ENDT`) is stable across
   one- and two-kernel outputs on current Xcode.

## Validation pipeline

Validation should be treated as a pipeline, not a single check:

1. Local parser (`air_validate`)
2. Apple CLI validation (`xcrun metallib --app-store-validate` or `xcrun metal -validate`)
3. Runtime load validation (`MTLDevice.newLibraryWithData:`)

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
container development and parser tests. The script `generate_reference_metallib.sh` falls back to
`tests/air_abi/reference/vector_add_air.ll` and generates
`tests/air_abi/reference/reference.experimental.metallib`.

The experimental container is useful for validating emitter/validator plumbing, but it is not an Apple
driver-compatible metallib.

## Next updates expected

- Compare this byte layout against additional Xcode builds and document field drift.
- Expand tag parsing beyond the currently consumed core set when new kernels require it.
- Keep MetalLibraryArchive validation wired in CI for parser cross-checks.
