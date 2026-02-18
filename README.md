CuMetal
=======

CuMetal is an experimental CUDA compiler/runtime for Apple Silicon GPUs.

Current code implements Phase 0.5 (metallib validation harness):

- `air_inspect`: inspect `.metallib` container structure
- `cumetal-air-emitter`: emit `.metallib` (xcrun mode or experimental mode)
- `air_validate`: structural checks and optional `xcrun metal -validate`
- `cumetal_metal_load_test`: check `MTLDevice.newLibraryWithData:` acceptance

This is bootstrap code for the compiler/runtime pipeline.
CUDA kernel execution is not implemented yet.

Build
-----

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Generate and validate a reference metallib (requires full Xcode)
-----------------------------------------------------------------

```bash
./scripts/generate_reference_metallib.sh
./build/air_inspect tests/air_abi/reference/reference.metallib
./build/air_validate tests/air_abi/reference/reference.metallib --xcrun
ctest --test-dir build -R air_abi_metal_load --output-on-failure
ctest --test-dir build -R air_abi_emit_validate_experimental --output-on-failure
```

If `xcrun metal`/`xcrun metallib` are unavailable
--------------------------------------------------

```bash
./build/cumetal-air-emitter \
  --input tests/air_abi/reference/vector_add_air.ll \
  --output /tmp/vector_add.experimental.metallib \
  --mode experimental \
  --overwrite

./build/air_validate /tmp/vector_add.experimental.metallib \
  --require-function-list --require-metadata
```

License
-------

[Apache 2.0](./LICENSE)
