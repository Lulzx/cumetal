CuMetal
=======

CuMetal is an experimental CUDA compiler/runtime for Apple Silicon GPUs.

Current status
--------------

Implemented today:

- Phase 0.5 tooling:
  - `air_inspect`: `.metallib` container inspection
  - `cumetal-air-emitter`: `.metallib` emission (xcrun-backed + experimental mode)
  - `air_validate`: structural checks + optional `xcrun metal -validate`
  - `cumetal_metal_load_test`: `MTLDevice.newLibraryWithData:` acceptance test
- Early Phase 0 runtime path:
  - allocation tracking (`ptr -> MTLBuffer`) with offset resolution
  - synchronous `cudaMemcpy` on UMA via `memcpy`
  - kernel launch through Metal compute pipelines (`setBuffer` + `setBytes`)
  - default-stream and user-stream execution (`cudaStreamCreate/Destroy/Synchronize`)
  - runtime functional tests for vector add and saxpy

Supported runtime API subset:

- `cudaInit`
- `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`
- `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaFree`
- `cudaHostAlloc`, `cudaFreeHost`
- `cudaMemcpy`, `cudaMemcpyAsync`
- `cudaMemset`, `cudaMemsetAsync`
- `cudaLaunchKernel`
- `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`
- `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamAddCallback`
- `cudaStreamWaitEvent`
- `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventRecord`
- `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy`
- `cudaDeviceSynchronize`
- `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorString`

Supported driver API subset:

- `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem`
- `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize`
- `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamAddCallback`, `cuStreamWaitEvent`
- `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventQuery`, `cuEventSynchronize`, `cuEventElapsedTime`
- `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, `cuModuleGetFunction`
- `cuLaunchKernel` (kernel params path and `extra` packed-argument path)
- `cuMemAlloc`, `cuMemFree`
- `cuMemAllocHost`, `cuMemFreeHost`
- `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`
- `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`
- `cuMemsetD8`, `cuMemsetD8Async`
- `cuGetErrorName`, `cuGetErrorString`

Current limitations:

- This is not yet a full CUDA Runtime/Driver implementation.
- Kernel launch currently uses a CuMetal descriptor (`cumetalKernel_t`) rather than NVCC fatbin registration.

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
ctest --test-dir build -R functional_runtime_host_alloc --output-on-failure
ctest --test-dir build -R functional_runtime_stream_flags --output-on-failure
ctest --test-dir build -R functional_runtime_stream_callback --output-on-failure
ctest --test-dir build -R functional_runtime_device_api --output-on-failure
ctest --test-dir build -R functional_runtime_device_properties --output-on-failure
ctest --test-dir build -R functional_driver_vector_add --output-on-failure
ctest --test-dir build -R functional_driver_device_api --output-on-failure
ctest --test-dir build -R functional_driver_device_query --output-on-failure
ctest --test-dir build -R functional_driver_stream_flags --output-on-failure
ctest --test-dir build -R functional_driver_stream_callback --output-on-failure
ctest --test-dir build -R functional_driver_context_switch --output-on-failure
ctest --test-dir build -R functional_driver_async_memcpy --output-on-failure
ctest --test-dir build -R functional_driver_memset --output-on-failure
ctest --test-dir build -R functional_driver_host_alloc --output-on-failure
ctest --test-dir build -R functional_driver_module_load_data --output-on-failure
ctest --test-dir build -R functional_driver_launch_extra --output-on-failure
ctest --test-dir build -R functional_driver_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_axpy_offset --output-on-failure
ctest --test-dir build -R unit_allocation_table --output-on-failure
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
