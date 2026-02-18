# CuMetal

CuMetal is a CUDA compiler/runtime project targeting Apple Silicon GPUs.

This repository currently implements the **Phase 0.5 metallib validation harness**:

- `air_inspect`
- `cumetal-air-emitter`
- `air_validate`
- `cumetal_metal_load_test`

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Generate + validate reference metallib

```bash
./scripts/generate_reference_metallib.sh
./build/air_inspect tests/air_abi/reference/reference.metallib
./build/air_validate tests/air_abi/reference/reference.metallib --xcrun
ctest --test-dir build -R air_abi_metal_load --output-on-failure
```

If `xcrun metal` / `xcrun metallib` are unavailable, use emitter experimental mode for
local parser/validator development:

```bash
./build/cumetal-air-emitter \
  --input tests/air_abi/reference/vector_add.metal \
  --output /tmp/vector_add.experimental.metallib \
  --mode experimental \
  --overwrite
```
