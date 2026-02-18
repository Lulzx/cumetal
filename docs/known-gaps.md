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
- PTX registration path now reaches parity acceptance for `llm.c` `test_gpt2_fp32.cu` in the CuMetal
  conformance harness (`OK (LOGITS)`, `LOSS OK`, `TENSOR OK`, `overall okay: 1`), but this is still
  mediated by the harness shim and tolerance patching (`CUMETAL_LLMC_GRAD_TOL`) rather than full
  instruction-accurate PTX->LLVM lowering for arbitrary kernels.
