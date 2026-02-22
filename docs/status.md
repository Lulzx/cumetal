# Status

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
  - PTX signature lowering now also covers unary `negate` and `reduce_sum` (atomic add) kernels
    used in regression tests for `neg.f32`, `shl.b64`, and `atom.global.add.f32` paths
  - intrinsic-lowering opcode coverage expanded for `div`, `rem`, `and`, `or`, `xor`, `not`,
    `selp`, and `rcp` instruction roots, with strict PTX sweep coverage
  - math intrinsic lowering extended: `fma`, `max/min/abs` (with float/int variants),
    `sqrt`, `rsqrt`, `ex2`→`exp2`, `lg2`→`log2`, `sin`, `cos`
  - warp primitive lowering: `shfl.sync.{idx,down,up,bfly}` → `air.simdgroup.shuffle*`,
    `vote.sync.{ballot,any,all}` → `air.simdgroup.{ballot,any,all}`,
    `bar.warp.sync` → `air.simdgroup.barrier` (__syncwarp emulation)
  - memory barrier lowering: `membar.gl/sys` → `air.mem.barrier.device`,
    `membar.cta` → `air.mem.barrier.threadgroup` (__threadfence/__threadfence_block)
  - async copy lowering: `cp.async.*` → `air.cp_async` (serialized ld+st);
    `cp.async.commit_group/wait_group/wait_all` → `air.threadgroup_barrier`
  - warp reduction lowering: `redux.sync.{add,and,or,xor,min,max}` →
    `air.simdgroup.reduce_{add,and,or,xor,min,max}[.f32]` (__redux_sync emulation)
  - parser: targeted error diagnostics for Hopper cluster ops (`cluster.*`, `mbarrier.*`),
    TMA (`cp.async.bulk.tensor.*`), and FP8 (`cvt.rn.f8*`) with specific messages
  - `cumetalc` accepts `.ptx` input via internal PTX->LLVM lowering (`--entry`, `--ptx-strict`)
  - `cumetalc` accepts initial `.cu` input via xcrun clang++ frontend lowering to LLVM IR
  - expanded PTX sweep harness (`tests/ptx_sweep`) for strict-mode supported/unsupported opcode checks
  - initial `intrinsic_lower` pass for thread-index/barrier/basic-math mappings
  - initial `printf_lower` pass for PTX `printf`/`vprintf` call extraction and format-table metadata
  - initial `addrspace` pass for shared/global/local load-store + `cvta.to.*` rewrites
  - initial `metadata` pass for AIR-style kernel metadata fields
  - initial phase1 pipeline API chaining parser + passes for a selected PTX entry
  - PTX parser handles entry attributes between signature/body (e.g. `.maxntid`, `.minnctapersm`)
    and `.param` qualifiers (`.ptr`, `.align`) used by clang-emitted PTX
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
  - cuBLAS `cublasSgemm`/`cublasSgemmStridedBatched` backed by MetalPerformanceShaders GEMM
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
  - llm.c FP32 CUDA stress binary can be built and executed through CuMetal registration path
    using `scripts/build_llmc_test_gpt2fp32cu.sh` + `scripts/run_llmc_test_gpt2fp32cu.sh`
  - `conformance_llmc_gpt2fp32cu` now enforces numerical parity markers and passes with
    `OK (LOGITS)`, `LOSS OK`, `TENSOR OK`, and `overall okay: 1`
  - llm.c harness build shim supports `CUMETAL_LLMC_GRAD_TOL` (default `1.2e-2`) to tune
    gradient-check tolerance applied to the generated test translation unit
  - llm.c runtime emulation fallback is now explicitly traceable (`CUMETAL_TRACE_LLMC_EMULATION=1`)
    and can be disabled (`CUMETAL_DISABLE_LLMC_EMULATION=1`) to validate pure PTX-lowered execution
  - direct Metal lowering for all 17 llm.c GPT-2 training kernels
    (`compiler/ptx/src/lower_to_metal.cpp`); `CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` now passes
    (`OK (LOGITS)`, `LOSS OK`, `TENSOR OK`, `overall okay: 1`) without any emulation fallback
  - PTX sweep extended with 30+ new test cases: `shfl.sync.{idx,down,up,bfly}`,
    `vote.sync.{ballot,any,all}`, `bar.warp.sync`, `membar.{gl,cta,sys}`,
    `cp.async.{ca,commit_group,wait_all}`, `redux.sync.{add,and,or,xor,min,max}`,
    and math intrinsics `sqrt`, `rsqrt`, `ex2`, `lg2`, `sin`, `cos`, `fma`, `abs`, `min`, `max`
  - Unsupported-op sweep extended with targeted diagnostic cases for Hopper cluster ops
    (`cluster.sync.aligned`, `mbarrier.init`, `mbarrier.arrive`), TMA
    (`cp.async.bulk.tensor.1d.*`), and FP8 (`cvt.rn.f8x2.*`)
  - `--fp64=native|emulate|warn` flag added to `cumetalc` (spec §8.1); `warn` mode emits
    per-instruction warnings for `.f64` opcodes; `emulate` implements Dekker FP32-pair
    decomposition for recognized fp64 kernels; runtime defaults to `kEmulate` because
    Apple Silicon GPU rejects `fmul double` in Metal pipelines at runtime (set
    `CUMETAL_FP64_MODE=native` to force native mode for compilation-path testing)
  - functional tests added:
    - `functional_runtime_warp_shuffle` (simd_shuffle broadcast, 64 threads, lane-0 broadcast)
    - `functional_runtime_fp16_ops` (half-precision add, 256 elements, exact integer check)
    - `functional_runtime_shared_reduce` (256-thread tree reduction, output[0]==256.0)
    - `functional_runtime_grid_2d` (4×4 grid of 2×2 blocks, linear index check)
    - `functional_runtime_grid_3d` (2×3×4 grid of 2×2×2 blocks, 3D linear index check)
    - `functional_runtime_fp64_ops` (PTX fma.rn.f64 via driver API; PASS via emulate mode)
    - `functional_runtime_atomic_shared` (threadgroup atomic, 128 blocks×256 threads=32768)
    - `functional_runtime_warp_vote` (simd_any/all/ballot; 64 threads, ballot=0x55555555)
    - `functional_runtime_struct_arg` (struct by-value argument via CUMETAL_ARG_BYTES)
    - `functional_runtime_barrier_order` (thread 0 writes sentinel; all threads verify post-barrier)
    - `functional_runtime_cp_async_emul` (cp.async emulated as ld+st+threadgroup_barrier)
    - `functional_runtime_warp_partial_mask` (spec §5.3/§10.3 partial-mask conservative lowering)
  - intrinsic lowering: `brev.b32/b64` → `llvm.bitreverse.i32/i64` added to pass and parser
  - intrinsic_lower unit tests: Test 6 (abs/shr), Test 7 (brev), Test 8 (f32/f64 math, b64 bitwise)
  - PTX sweep: expanded to 93+ cases covering all kSupportedRoots opcode roots including:
    - `clz.b64`, `popc.b64` (64-bit bit-count ops)
    - `add/sub/mul/div.f32` (basic float arithmetic)
    - `neg/abs/min/max.f64` (double-precision unary/binary)
    - `and/or/xor/not.b64` (64-bit bitwise ops)
    - `mul.lo.u64`, `rem.u32`, `rem.s64`
    - `abs.{s32,s64,f32,f64}`, `shr.{b32,u32,s32,b64,u64,s64}`
    - `vote.{ballot,any,all}` non-sync forms
    - `st.global.{u32,u64,f64}`, `ld.global.{u8,s8,u16,s16}`
    - `atom.global.{cas,and,or,xor,min,max,exch}.b32`
    - `redux.sync.{min,max}.f32`
    - partial-mask variants: `shfl.sync` and `vote.sync.ballot` with mask=0x0000FFFF

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
- `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaFuncSetSharedMemConfig`, `cudaFuncSetAttribute`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`
- `cudaPointerGetAttributes`, `cudaChooseDevice`
- `cudaStreamCreateWithPriority` (priority ignored; creates regular stream)
- `cudaDeviceSetLimit` (no-op), `cudaDeviceGetLimit` (returns sensible defaults)
- `cudaLaunchCooperativeKernel` (forwards to `cudaLaunchKernel`; threadgroup CG works)
- `cudaDeviceSetCacheConfig`, `cudaDeviceGetCacheConfig` (no-op stubs; all memory is UMA)
- `cudaDeviceSetSharedMemConfig`, `cudaDeviceGetSharedMemConfig` (no-op stubs)
- `cudaGetSymbolAddress`, `cudaGetSymbolSize`
- `cudaMemPrefetchAsync`, `cudaMemAdvise`, `cudaMemRangeGetAttribute` (meaningful no-ops on Apple Silicon UMA)

`cudaDeviceProp` fields now populated per spec §6.8:
- `unifiedAddressing = 1`, `managedMemory = 1`, `concurrentManagedAccess = 1` (UMA)
- `maxBufferArguments = 31` (Metal buffer argument limit)

`cudaDeviceGetAttribute` and `cuDeviceGetAttribute` now support additional attributes:
- `cudaDevAttrComputeCapabilityMajor` / `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` → 8
- `cudaDevAttrComputeCapabilityMinor` / `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` → 0
- `cudaDevAttrMaxRegistersPerBlock` / `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK` → 65536
- `cudaDevAttrClockRate` / `CU_DEVICE_ATTRIBUTE_CLOCK_RATE` → 1296000 kHz
- `cudaDevAttrTextureAlignment` → 512 bytes
- `cudaDevAttrGpuOverlap` / `CU_DEVICE_ATTRIBUTE_GPU_OVERLAP` → 1

`cooperative_groups::thread_block_tile<N>` extended with:
- `shfl(val, src_rank)`, `shfl_down(val, delta)`, `shfl_xor(val, mask)`
- `any(pred)`, `all(pred)`, `ballot(pred)` (via `__nvvm_vote_*` builtins)
- `cooperative_groups::less<T>` binary operator alongside existing `plus<T>` and `greater<T>`

CUDA vector types added to `cuda_runtime.h`:
- All standard types: `char2/3/4`, `short2/3/4`, `int2/3/4`, `uint2/4`,
  `long2/4`, `longlong2/4`, `ulong2/4`, `ulonglong2/4`,
  `float2/3`, `double2/3/4` with `__align__` annotations and `make_*` constructors

Device atomics added (CUDA device code path, spec §6.7):
- `atomicSub`, `atomicExch` (int/uint/float), `atomicMin`/`atomicMax` (int/uint),
  `atomicCAS` (uint/int/ull), `atomicAnd`/`atomicOr`/`atomicXor` (int/uint)

Device intrinsics added (guarded by `#ifndef __CLANG_CUDA_DEVICE_FUNCTIONS_H__`):
- `__syncwarp`, `__threadfence`, `__threadfence_block`, `__threadfence_system`
- `__activemask`, `__popc`/`__popcll`, `__clz`/`__clzll`, `__brev`/`__brevll`
- `__ffs`/`__ffsll`, `__fmaf_rn`, `__fma_rn`

`install.sh` / `uninstall.sh` now detect fish shell (`$SHELL=*/fish`) and write
`set -gx` syntax to `~/.config/fish/config.fish`; `CUMETAL_SHELL_RC` overrides.

Supported driver API subset:

- `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute`
- `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxGetFlags`, `cuCtxSetFlags`, `cuCtxSynchronize`
- `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamAddCallback`, `cuStreamWaitEvent`
- `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventQuery`, `cuEventSynchronize`, `cuEventElapsedTime`
- `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, `cuModuleGetFunction`
- `cuModuleLoadData` accepts metallib bytes/paths and PTX text images (including basic CUDA fatbin wrapper PTX variants)
- `cuModuleLoadDataEx` accepts option arrays in compatibility mode (options are currently ignored)
- `cuLaunchKernel` (kernel params path and `extra` packed-argument path)
- `cuMemAlloc`, `cuMemAllocManaged`, `cuMemFree`
- `cuMemGetInfo`
- `cuMemAllocHost`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`, `cuMemFreeHost`
- `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`
- `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`
- `cuMemsetD8`, `cuMemsetD8Async`
- `cuGetErrorName`, `cuGetErrorString`
- `cuProfilerStart`, `cuProfilerStop`
- `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize`
- `cuFuncGetAttribute`, `cuFuncSetCacheConfig`
- `cuStreamCreateWithPriority` (priority ignored; creates regular stream)
- `cuLaunchCooperativeKernel` (forwards to `cuLaunchKernel`; threadgroup CG works)
- `cuMemsetD16`, `cuMemsetD32`, `cuMemsetD16Async`, `cuMemsetD32Async`
- `cuDeviceComputeCapability` (returns 8.0 — synthetic Ampere-equivalent)
- `cuDeviceCanAccessPeer` (returns 0; single GPU on Apple Silicon)

Public headers now installed: `cuda.h`, `cuda_runtime.h`, `cuda_fp16.h`, `cublas_v2.h`,
`cufft.h`, `curand.h`, `cooperative_groups.h`, `cooperative_groups/reduce.h`.
`cuda_fp16.h` provides host-side `__half` (IEEE 754 float16 via bit manipulation) and
device-side `__half = _Float16`; `atomicAdd(__half*, __half)` via CAS loop (spec §8).

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
- cuFFT (`cufft.h`)
  - `cufftCreate`, `cufftDestroy`, `cufftSetStream`, `cufftGetSize`, `cufftGetVersion`
  - `cufftPlan1d`, `cufftPlan2d`, `cufftPlan3d`, `cufftPlanMany`
  - `cufftMakePlan1d`, `cufftMakePlan2d`, `cufftMakePlan3d`, `cufftMakePlanMany`
  - `cufftExecC2C`, `cufftExecR2C`, `cufftExecC2R` (single-precision)
  - `cufftExecZ2Z`, `cufftExecD2Z`, `cufftExecZ2D` (double-precision)
  - Backed by Apple Accelerate `vDSP_DFT_Execute` (arbitrary N, any batch size)
  - `libcufft.dylib` symlink alias to `libcumetal.dylib`
- cuBLAS v2 (`cublas_v2.h`)
  - `cublasCreate`, `cublasDestroy`, `cublasGetVersion`
  - `cublasSetStream`, `cublasGetStream`
  - `cublasSetMathMode`, `cublasGetMathMode`
  - `cublasSaxpy`, `cublasSscal`, `cublasScopy`, `cublasSgemm`
  - `cublasSgemmStridedBatched`, `cublasDgemmStridedBatched`
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
  fatbin PTX images (wrapper and direct blob) are supported (including `FatBinary2/FatBinary3` entry
  points), but full NVCC fatbinary variants are not yet implemented.
