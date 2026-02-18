CuMetal
=======

CuMetal is an experimental CUDA compiler/runtime for Apple Silicon GPUs.

Current status
--------------

Implemented today:

- Phase 0.5 tooling:
  - `air_inspect`: `.metallib` container inspection
  - `cumetal-air-emitter`: `.metallib` emission (xcrun-backed + experimental mode)
  - `cumetalc`: thin compiler-driver CLI over the AIR emitter
  - `air_validate`: structural checks + optional `xcrun metal -validate`
  - `cumetal_metal_load_test`: `MTLDevice.newLibraryWithData:` acceptance test
- Early Phase 0 runtime path:
  - allocation tracking (`ptr -> MTLBuffer`) with offset resolution
  - synchronous `cudaMemcpy` on UMA via `memcpy`
  - kernel launch through Metal compute pipelines (`setBuffer` + `setBytes`)
  - default-stream and user-stream execution (`cudaStreamCreate/Destroy/Synchronize`)
  - runtime functional tests for vector add and saxpy
  - driver module loading from both in-memory metallib bytes and filesystem paths
  - on-disk cache for `cuModuleLoadData` metallib byte payloads
  - driver stream/event/memory APIs enforce `cuInit` + current-context requirements
  - shared runtime artifact: `libcumetal.dylib` (plus `cuda.h` / `cuda_runtime.h` install headers)
  - startup conflict warning if another `libcuda.dylib` is already loaded
  - Metal command-buffer failures map to CUDA timeout/illegal-address/devices-unavailable errors
  - default module cache root: `$HOME/Library/Caches/io.cumetal/kernels` (override: `CUMETAL_CACHE_DIR`)

Supported runtime API subset:

- `cudaInit`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`
- `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`
- `cudaSetDeviceFlags`, `cudaGetDeviceFlags`
- `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaFree`
- `cudaHostAlloc`, `cudaFreeHost`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`
- `cudaMemGetInfo`
- `cudaMemcpy`, `cudaMemcpyAsync`
- `cudaMemset`, `cudaMemsetAsync`
- `cudaLaunchKernel`
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
- `cuLaunchKernel` (kernel params path and `extra` packed-argument path)
- `cuMemAlloc`, `cuMemAllocManaged`, `cuMemFree`
- `cuMemGetInfo`
- `cuMemAllocHost`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`, `cuMemFreeHost`
- `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`
- `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`
- `cuMemsetD8`, `cuMemsetD8Async`
- `cuGetErrorName`, `cuGetErrorString`
- `cuProfilerStart`, `cuProfilerStop`

Current limitations:

- This is not yet a full CUDA Runtime/Driver implementation.
- Kernel launch currently uses a CuMetal descriptor (`cumetalKernel_t`) rather than NVCC fatbin registration.

Build
-----

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cmake --install build --prefix /tmp/cumetal-install
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
ctest --test-dir build -R air_abi_metal_load --output-on-failure
ctest --test-dir build -R air_abi_emit_validate_experimental --output-on-failure
ctest --test-dir build -R air_abi_validate_negative --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_positional_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_default_output_emit_load_xcrun --output-on-failure
```

Runtime execution tests
-----------------------

These tests compile Metal kernels with `xcrun` and run them through the CuMetal runtime:

```bash
ctest --test-dir build -R functional_runtime_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_stream_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_async_memops --output-on-failure
ctest --test-dir build -R functional_runtime_event --output-on-failure
ctest --test-dir build -R functional_runtime_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_stream_query --output-on-failure
ctest --test-dir build -R functional_runtime_memcpy_kind --output-on-failure
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
ctest --test-dir build -R functional_driver_vector_add --output-on-failure
ctest --test-dir build -R functional_driver_device_api --output-on-failure
ctest --test-dir build -R functional_driver_error_api --output-on-failure
ctest --test-dir build -R functional_driver_profiler_api --output-on-failure
ctest --test-dir build -R functional_driver_device_query --output-on-failure
ctest --test-dir build -R functional_driver_device_attribute --output-on-failure
ctest --test-dir build -R functional_driver_stream_flags --output-on-failure
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
ctest --test-dir build -R functional_driver_launch_extra --output-on-failure
ctest --test-dir build -R functional_driver_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_axpy_offset --output-on-failure
ctest --test-dir build -R unit_allocation_table --output-on-failure
ctest --test-dir build -R unit_module_cache --output-on-failure
ctest --test-dir build -R unit_library_conflict --output-on-failure
ctest --test-dir build -R unit_install_uninstall_scripts --output-on-failure
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
