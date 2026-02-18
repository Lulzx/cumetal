# Known Gaps

- Full AIR metadata reverse-engineering is still in progress.
- `cumetal-air-emitter --mode experimental` emits a CuMetal test container, not Apple's metallib ABI.
- MetalLibraryArchive validation is optional and currently executed through an external bridge command.
- Binary-shim registration currently supports CuMetal envelopes (`CMTL`) plus basic CUDA fatbin-wrapper PTX
  images and host-function mapping, but not full NVCC fatbinary variants.
- `llm.c` stress coverage is wired as an opt-in harness (`conformance_llmc_gpt2fp32cu`) and depends on an
  external checkout/build command.
