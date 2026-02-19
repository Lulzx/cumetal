# Known Gaps

- Full AIR metadata reverse-engineering is still in progress.
- `cumetal-air-emitter --mode experimental` emits a CuMetal test container, not Apple's metallib ABI.
- MetalLibraryArchive validation is optional and currently executed through an external bridge command.
- Binary-shim registration supports CuMetal envelopes (`CMTL`), direct PTX images, and basic CUDA
  fatbin PTX images (wrapper and direct blob) with `FatBinary/FatBinary2/FatBinary3` entry points, plus
  host-function and basic symbol-variable mapping, but not full NVCC fatbinary variants.
- `llm.c` stress coverage (`conformance_llmc_gpt2fp32cu`) now auto-builds/runs via CuMetal's local
  clang+fatbin shim scripts when an `llm.c` checkout is present, but this path still depends on an external
  checkout with model/debug-state assets.
- End-to-end GPU execution of full `llm.c` CUDA sources through `cumetalc` is not yet implemented:
  current `.cu` frontend lowering is partial and does not support full CUDA language/device-runtime
  semantics required by `train_gpt2_fp32.cu` (cooperative groups, CUDA builtins, and kernel-launch codegen).
- PTX registration path reaches full parity for `llm.c` `test_gpt2_fp32.cu` via direct Metal lowering
  for all 17 GPT-2 training kernels. `CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` now passes (`OK (LOGITS)`,
  `LOSS OK`, `TENSOR OK`, `overall okay: 1`) without the emulation fallback. The direct Metal lowering
  path (`compiler/ptx/src/lower_to_metal.cpp`) is a hardcoded set matched by kernel name; kernels not
  in that set fall back to PTX→LLVM lowering. A generalized PTX→Metal instruction-level translator
  (not name-matched) remains an open Phase 4 deliverable for broader kernel coverage.
