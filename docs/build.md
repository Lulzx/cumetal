# Build and Validation

Build
-----

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cmake --install build --prefix /tmp/cumetal-install
# optional: enable binary-shim registration exports + install libcuda.dylib alias
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCUMETAL_ENABLE_BINARY_SHIM=ON
```

Generate and validate a reference metallib (requires full Xcode)
-----------------------------------------------------------------

```bash
./scripts/generate_reference_metallib.sh
./build/air_inspect tests/air_abi/reference/reference.metallib
./build/air_validate tests/air_abi/reference/reference.metallib --xcrun
./build/cumetalc --mode xcrun --input tests/air_abi/reference/vector_add.metal --output /tmp/vector_add.cumetalc.metallib --overwrite
./build/cumetalc --mode xcrun tests/air_abi/reference/vector_add.metal -o /tmp/vector_add.cumetalc.positional.metallib --overwrite
./build/cumetalc --mode xcrun tests/air_abi/reference/vector_add.metal --overwrite
./build/cumetalc --mode experimental --input tests/air_abi/reference/vector_add.cu --output /tmp/vector_add.cumetalc.from_cu.experimental.metallib --overwrite
./build/cumetal-ptx2llvm --input tests/air_abi/reference/vector_add.ptx --output /tmp/vector_add.from_ptx.ll --entry vector_add --overwrite
./build/cumetal-ptx2llvm tests/air_abi/reference/vector_add.ptx --entry vector_add --overwrite
ctest --test-dir build -R air_abi_metal_load --output-on-failure
ctest --test-dir build -R air_abi_emit_validate_experimental --output-on-failure
ctest --test-dir build -R air_abi_validate_negative --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_positional_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_default_output_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_multikernel_emit_validate_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_ptx_to_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_matrix_ptx_to_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_ptx_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_matrix_ptx_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_cu_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_cu_default_output_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_cu_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_ptx_default_output_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_ptx_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_matrix_ptx_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_ptx2llvm_positional_default_output --output-on-failure
ctest --test-dir build -R air_abi_xcode_matrix_regression --output-on-failure
```

Optional Xcode 15/16 ABI matrix setup:

```bash
export CUMETAL_XCODE15_DEVELOPER_DIR="/Applications/Xcode_15.app/Contents/Developer"
export CUMETAL_XCODE16_DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
```

