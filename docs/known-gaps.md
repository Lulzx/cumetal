# Known Gaps

- Full AIR metadata reverse-engineering is still in progress.
- `cumetal-air-emitter --mode experimental` emits a CuMetal test container, not Apple's metallib ABI.
- MetalLibraryArchive validation is optional and currently executed through an external bridge command.
- Binary-shim registration supports CuMetal envelopes (`CMTL`), direct PTX images, and basic CUDA
  fatbin PTX images (wrapper and direct blob) with `FatBinary/FatBinary2/FatBinary3` entry points, plus
  host-function and basic symbol-variable mapping, but not full NVCC fatbinary variants.
- Warp-primitive intrinsic lowering (`shfl.sync.*`, `vote.sync.{ballot,any,all}`, `bar.warp.sync`)
  is implemented in the `intrinsic_lower` pass, mapping to `air.simdgroup.{shuffle,shuffle_down,
  shuffle_up,shuffle_xor,ballot,any,all,barrier}`. Mask emulation for non-0xFFFFFFFF membermasks is
  conservative (full-group barrier) rather than lane-selective; kernels using partial masks should be
  tested carefully. The generic PTX→Metal emitter does not yet emit Metal `simd_shuffle*` calls for
  these instructions — they require the LLVM→AIR path.
- `llm.c` stress coverage (`conformance_llmc_gpt2fp32cu`) now auto-builds/runs via CuMetal's local
  clang+fatbin shim scripts when an `llm.c` checkout is present, but this path still depends on an external
  checkout with model/debug-state assets.
- End-to-end GPU execution of full `llm.c` CUDA sources through `cumetalc` is not yet implemented:
  current `.cu` frontend lowering is partial and does not support full CUDA language/device-runtime
  semantics required by `train_gpt2_fp32.cu` (cooperative groups, CUDA builtins, and kernel-launch codegen).
- PTX registration path reaches full parity for `llm.c` `test_gpt2_fp32.cu` via direct Metal lowering
  for all 17 GPT-2 training kernels. `CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` now passes without any
  emulation fallback. The lowering path (`compiler/ptx/src/lower_to_metal.cpp`) has two tiers:
  (1) hardcoded name-matched kernels for the 17 llm.c GPT-2 kernels, and (2) a generic
  PTX→Metal instruction-level translator that handles arbitrary element-wise kernels via two-pass
  register-provenance analysis. The generic emitter supports: `ld/st.global`, `atom.global.add.f32`,
  `setp`-based bounds guards, `mad.lo.u32` and `mul.lo.u32`+`add.u32` GID patterns, arithmetic
  (`add`, `sub`, `mul`, `div`, `rem`, `shl`, `shr`, `and`, `or`, `xor`, `not`, `selp`), `fma`/`mad`,
  `neg`/`abs`/`rcp`, `max`/`min`, and the unary math intrinsics `sqrt`, `rsqrt`, `ex2`→`exp2`,
  `lg2`→`log2`, `sin`, `cos`. Kernels with unsupported patterns fall back to PTX→LLVM lowering.
- `--fp64=emulate` mode (Dekker's algorithm FP32-pair decomposition, spec §8.1) is not yet
  implemented. The flag is accepted and parsed, but currently falls through to native FP64 behavior
  (`--fp64=native`). A diagnostic warning is emitted to alert the user. Implementing Dekker's
  decomposition requires a dedicated LLVM pass that is deferred to a future milestone.
- Null stream synchronization (spec §6.3.1) is implemented via command-buffer sequencing on the
  default `MTLCommandQueue` rather than the spec's described `MTLSharedEvent`-based approach.
  The observable CUDA semantics (null-stream serialization) are correct for the common single-context
  use case, but the explicit multi-stream "all user streams wait for null stream" guarantee is not
  fully enforced via events. `MTLSharedEvent` integration is deferred to a future milestone.
- Device printf (spec §5.3): the `printf_lower` compiler pass extracts format strings and emits
  metadata, but the runtime buffer allocation, binding, and post-kernel drain (spec §6.5 step 10)
  are not yet implemented. Device `printf` calls in kernels will be silently dropped at runtime.
