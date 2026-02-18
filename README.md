CuMetal
=======

CuMetal is an experimental CUDA compiler/runtime for Apple Silicon GPUs.

Current status
--------------

Implemented today:

- Phase 0.5 tooling:
  - `air_inspect`: `.metallib` container inspection
    - parses Apple function-list tags (`NAME`/`TYPE`/`HASH`/`MDSZ`/`OFFT`/`VERS`) on current Xcode layout
  - `cumetal-air-emitter`: `.metallib` emission (xcrun-backed + experimental mode)
  - `cumetalc`: thin compiler-driver CLI over the AIR emitter
  - `air_validate`: structural checks + optional `xcrun metal -validate`
  - `cumetal_metal_load_test`: `MTLDevice.newLibraryWithData:` acceptance test
- Phase 1 scaffolding:
  - minimal PTX text parser (`.version` / `.target` / `.entry` / `.param` + instruction stream)
    with tolerant/strict unsupported-op modes in `compiler/ptx/`
  - `cumetal-ptx2llvm`: PTX text to LLVM IR (AIR metadata scaffold) via the phase1 pipeline,
    including concrete vector-add and matrix-multiply body emission for recognized signatures
  - `cumetalc` accepts `.ptx` input via internal PTX->LLVM lowering (`--entry`, `--ptx-strict`)
  - `cumetalc` accepts initial `.cu` input via xcrun clang++ frontend lowering to LLVM IR
  - expanded PTX sweep harness (`tests/ptx_sweep`) for strict-mode supported/unsupported opcode checks
  - initial `intrinsic_lower` pass for thread-index/barrier/basic-math mappings
  - initial `printf_lower` pass for PTX `printf`/`vprintf` call extraction and format-table metadata
  - initial `addrspace` pass for shared/global/local load-store + `cvta.to.*` rewrites
  - initial `metadata` pass for AIR-style kernel metadata fields
  - initial phase1 pipeline API chaining parser + passes for a selected PTX entry
- Early Phase 0 runtime path:
  - allocation tracking (`ptr -> MTLBuffer`) with offset resolution
  - optional `MTLHeap`-backed sub-allocation path for `cudaMalloc` / `cuMemAlloc`
    (`CUMETAL_MTLHEAP_ALLOC=1`, chunk size override: `CUMETAL_MTLHEAP_CHUNK_BYTES`)
  - synchronous `cudaMemcpy` on UMA via `memcpy`
  - kernel launch through Metal compute pipelines (`setBuffer` + `setBytes`)
  - default-stream, per-thread default stream, and user-stream execution
    (`cudaStreamCreate/Destroy/Synchronize`, `cudaStreamPerThread`, `cudaStreamLegacy`)
  - runtime functional tests for vector add, matrix multiply, and saxpy
  - initial library shims for cuRAND and cuBLAS v2
  - driver module loading from both in-memory metallib bytes and filesystem paths
  - on-disk cache for `cuModuleLoadData` metallib byte payloads
  - driver stream/event/memory APIs enforce `cuInit` + current-context requirements
  - shared runtime artifact: `libcumetal.dylib` (plus `cuda.h` / `cuda_runtime.h` install headers)
  - startup conflict warning if another `libcuda.dylib` is already loaded
  - Metal command-buffer failures map to CUDA timeout/illegal-address/devices-unavailable errors
  - default module cache root: `$HOME/Library/Caches/io.cumetal/kernels` (override: `CUMETAL_CACHE_DIR`)
  - `samples/vectorAdd` source flow exercised end-to-end (compile `.cu` with `cumetalc`, link host app
    against `libcumetal`, execute and validate output)
  - opt-in registration path symbols for binary-shim style launches
    (`__cudaRegisterFatBinary`, `__cudaRegisterFatBinary2`, `__cudaRegisterFatBinary3`,
    `__cudaRegisterFatBinaryEnd`, `__cudaRegisterFunction`, `__cudaRegisterVar`,
    `__cudaRegisterManagedVar`,
    `__cudaPushCallConfiguration`)
  - legacy runtime launch path (`cudaConfigureCall` / `cudaSetupArgument` / `cudaLaunch`)

Supported runtime API subset:

- `cudaInit`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`
- `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`
- `cudaSetDeviceFlags`, `cudaGetDeviceFlags`
- `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaFree`
- `cudaHostAlloc`, `cudaFreeHost`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`
- `cudaMemGetInfo`
- `cudaMemcpy`, `cudaMemcpyAsync`
- `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbolAsync`
- `cudaMemset`, `cudaMemsetAsync`
- `cudaLaunchKernel`
- `cudaConfigureCall`, `cudaSetupArgument`, `cudaLaunch`
- `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`
- `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamAddCallback`
- `cudaStreamWaitEvent`
- `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventRecord`
- `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy`
- `cudaDeviceReset`
- `cudaDeviceSynchronize`
- `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`, `cudaGetErrorString`
- `cudaProfilerStart`, `cudaProfilerStop`

Supported driver API subset:

- `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute`
- `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxGetFlags`, `cuCtxSetFlags`, `cuCtxSynchronize`
- `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamAddCallback`, `cuStreamWaitEvent`
- `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventQuery`, `cuEventSynchronize`, `cuEventElapsedTime`
- `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, `cuModuleGetFunction`
- `cuModuleLoadData` accepts metallib bytes/paths and PTX text images (including basic CUDA fatbin wrapper PTX variants)
- `cuLaunchKernel` (kernel params path and `extra` packed-argument path)
- `cuMemAlloc`, `cuMemAllocManaged`, `cuMemFree`
- `cuMemGetInfo`
- `cuMemAllocHost`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`, `cuMemFreeHost`
- `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`
- `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`
- `cuMemsetD8`, `cuMemsetD8Async`
- `cuGetErrorName`, `cuGetErrorString`
- `cuProfilerStart`, `cuProfilerStop`

Supported library shim subset:

- cuRAND (`curand.h`)
  - `curandCreateGenerator`, `curandDestroyGenerator`
  - `curandGetVersion`
  - `curandSetStream`, `curandGetStream`
  - `curandSetPseudoRandomGeneratorSeed`, `curandSetGeneratorOffset`
  - `curandGenerate` (uint32 output), `curandGenerateLongLong` (uint64 output)
  - `curandGenerateUniform`, `curandGenerateUniformDouble`
  - `curandGenerateNormal`, `curandGenerateNormalDouble`
  - `curandGenerateLogNormal`, `curandGenerateLogNormalDouble`
- cuBLAS v2 (`cublas_v2.h`)
  - `cublasCreate`, `cublasDestroy`, `cublasGetVersion`
  - `cublasSetStream`, `cublasGetStream`
  - `cublasSaxpy`, `cublasSscal`, `cublasScopy`, `cublasSgemm`
  - `cublasSswap`, `cublasDswap`
  - `cublasSdot`, `cublasDdot`
  - `cublasSasum`, `cublasDasum`
  - `cublasSnrm2`, `cublasDnrm2`
  - `cublasIsamax`, `cublasIdamax`
  - `cublasIsamin`, `cublasIdamin`
  - `cublasSgemv`, `cublasDgemv`
  - `cublasSger`, `cublasDger`
  - `cublasSsymv`, `cublasDsymv`
  - `cublasDaxpy`, `cublasDscal`, `cublasDcopy`, `cublasDgemm`

Library alias compatibility:

- Build/install also provides `libcublas.dylib` and `libcurand.dylib` aliases to
  `libcumetal.dylib`, so software linked against CUDA library names can resolve shim symbols.
- Optional binary-shim alias: when `CUMETAL_ENABLE_BINARY_SHIM=ON`, build/install also provides
  `libcuda.dylib -> libcumetal.dylib`.

Current limitations:

- This is not yet a full CUDA Runtime/Driver implementation.
- Default kernel launch uses a CuMetal descriptor (`cumetalKernel_t`).
- Binary-shim registration is partial: CuMetal `CMTL` envelopes, direct PTX images, and basic CUDA
  fatbin-wrapper PTX images are supported (including `FatBinary2/FatBinary3` entry points), but full NVCC
  fatbinary variants are not yet implemented.

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

Runtime execution tests
-----------------------

These tests compile Metal kernels with `xcrun` and run them through the CuMetal runtime:

```bash
ctest --test-dir build -R functional_runtime_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_vector_add_heap_alloc --output-on-failure
ctest --test-dir build -R functional_runtime_vector_add_cu --output-on-failure
ctest --test-dir build -R functional_sample_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_matrix_mul --output-on-failure
ctest --test-dir build -R functional_runtime_stream_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_null_stream_sync --output-on-failure
ctest --test-dir build -R functional_runtime_stream_per_thread --output-on-failure
ctest --test-dir build -R functional_runtime_async_memops --output-on-failure
ctest --test-dir build -R functional_runtime_event --output-on-failure
ctest --test-dir build -R functional_runtime_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_stream_query --output-on-failure
ctest --test-dir build -R functional_runtime_memcpy_kind --output-on-failure
ctest --test-dir build -R functional_runtime_symbol_memcpy --output-on-failure
ctest --test-dir build -R functional_runtime_mem_get_info --output-on-failure
ctest --test-dir build -R functional_runtime_host_alloc --output-on-failure
ctest --test-dir build -R functional_runtime_host_pointer_api --output-on-failure
ctest --test-dir build -R functional_runtime_stream_flags --output-on-failure
ctest --test-dir build -R functional_runtime_stream_callback --output-on-failure
ctest --test-dir build -R functional_runtime_device_api --output-on-failure
ctest --test-dir build -R functional_runtime_device_properties --output-on-failure
ctest --test-dir build -R functional_runtime_device_attribute --output-on-failure
ctest --test-dir build -R functional_runtime_device_reset --output-on-failure
ctest --test-dir build -R functional_runtime_device_flags --output-on-failure
ctest --test-dir build -R functional_runtime_error_api --output-on-failure
ctest --test-dir build -R functional_runtime_profiler_api --output-on-failure
ctest --test-dir build -R functional_curand_uniform --output-on-failure
ctest --test-dir build -R functional_cublas_api --output-on-failure
ctest --test-dir build -R functional_driver_vector_add --output-on-failure
ctest --test-dir build -R functional_driver_matrix_mul --output-on-failure
ctest --test-dir build -R functional_driver_null_stream_sync --output-on-failure
ctest --test-dir build -R functional_driver_device_api --output-on-failure
ctest --test-dir build -R functional_driver_error_api --output-on-failure
ctest --test-dir build -R functional_driver_profiler_api --output-on-failure
ctest --test-dir build -R functional_driver_device_query --output-on-failure
ctest --test-dir build -R functional_driver_device_attribute --output-on-failure
ctest --test-dir build -R functional_driver_stream_flags --output-on-failure
ctest --test-dir build -R functional_driver_stream_per_thread --output-on-failure
ctest --test-dir build -R functional_driver_stream_callback --output-on-failure
ctest --test-dir build -R functional_driver_context_switch --output-on-failure
ctest --test-dir build -R functional_driver_context_requirements --output-on-failure
ctest --test-dir build -R functional_driver_async_memcpy --output-on-failure
ctest --test-dir build -R functional_driver_memset --output-on-failure
ctest --test-dir build -R functional_driver_mem_get_info --output-on-failure
ctest --test-dir build -R functional_driver_mem_alloc_managed --output-on-failure
ctest --test-dir build -R functional_driver_host_alloc --output-on-failure
ctest --test-dir build -R functional_driver_host_pointer_api --output-on-failure
ctest --test-dir build -R functional_driver_module_load_data --output-on-failure
ctest --test-dir build -R functional_driver_module_load_data_ptx --output-on-failure
ctest --test-dir build -R functional_driver_launch_extra --output-on-failure
ctest --test-dir build -R functional_driver_launch_extra_scalar --output-on-failure
ctest --test-dir build -R functional_driver_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_axpy_offset --output-on-failure
ctest --test-dir build -R functional_runtime_atomic --output-on-failure
# binary-shim-only tests (`CUMETAL_ENABLE_BINARY_SHIM=ON`):
ctest --test-dir build -R functional_runtime_registration_path --output-on-failure
ctest --test-dir build -R functional_runtime_call_config_registration --output-on-failure
ctest --test-dir build -R functional_runtime_registration_fatbin_ptx --output-on-failure
ctest --test-dir build -R functional_runtime_legacy_launch_registration --output-on-failure
ctest --test-dir build -R functional_runtime_registration_fatbinary2_symbols --output-on-failure
ctest --test-dir build -R functional_runtime_registration_var_symbol --output-on-failure
ctest --test-dir build -R unit_allocation_table --output-on-failure
ctest --test-dir build -R unit_module_cache --output-on-failure
ctest --test-dir build -R unit_library_conflict --output-on-failure
ctest --test-dir build -R unit_metallib_parser --output-on-failure
ctest --test-dir build -R unit_ptx_parser --output-on-failure
ctest --test-dir build -R unit_intrinsic_lower --output-on-failure
ctest --test-dir build -R unit_printf_lower --output-on-failure
ctest --test-dir build -R unit_addrspace_pass --output-on-failure
ctest --test-dir build -R unit_metadata_pass --output-on-failure
ctest --test-dir build -R unit_phase1_pipeline --output-on-failure
ctest --test-dir build -R unit_ptx_lower_to_llvm --output-on-failure
ctest --test-dir build -R unit_cumetal_bench_help --output-on-failure
ctest --test-dir build -R unit_cumetal_bench_ratio_gate --output-on-failure
ctest --test-dir build -R unit_runtime_library_aliases --output-on-failure
# binary-shim-only unit tests (`CUMETAL_ENABLE_BINARY_SHIM=ON`):
ctest --test-dir build -R unit_binary_shim_symbol_exports --output-on-failure
ctest --test-dir build -R unit_binary_shim_library_alias --output-on-failure
ctest --test-dir build -R unit_binary_shim_link_alias --output-on-failure
ctest --test-dir build -R unit_library_link_aliases --output-on-failure
ctest --test-dir build -R ptx_sweep_supported_ops --output-on-failure
ctest --test-dir build -R ptx_sweep_unsupported_ops --output-on-failure
ctest --test-dir build -R unit_install_uninstall_scripts --output-on-failure
```

Conformance suite
-----------------

Phase 4 conformance gate over functional tests:

```bash
ctest --test-dir build -R conformance_phase4_functional --output-on-failure
ctest --test-dir build -R conformance_llmc_gpt2fp32cu --output-on-failure
```

Direct invocation with custom threshold/regex:

```bash
./tests/conformance/run_conformance_suite.sh build 90 '^functional_'
```

Optional llm.c stress harness setup:

```bash
export CUMETAL_LLMC_DIR="/path/to/llm.c"
# optional:
export CUMETAL_LLMC_BUILD_CMD="make test_gpt2fp32cu"
export CUMETAL_LLMC_TEST_CMD="./test_gpt2fp32cu"
```

Benchmark runner
----------------

```bash
./scripts/generate_reference_metallib.sh
./build/cumetal_bench \
  --metallib tests/air_abi/reference/reference.metallib \
  --kernel vector_add \
  --elements 262144 \
  --warmup 5 \
  --iterations 50 \
  --max-ratio 2.0
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
