# Known Gaps

- Full AIR metadata reverse-engineering is still in progress.
- `cumetal-air-emitter --mode experimental` emits a CuMetal test container, not Apple's metallib ABI.
- MetalLibraryArchive validation is optional and currently executed through an external bridge command.
- Binary-shim registration currently supports a CuMetal envelope (`CMTL` fatbin image) and host-function
  mapping, not full NVCC fatbinary parsing.
- `llm.c` stress coverage is wired as an opt-in harness (`conformance_llmc_gpt2fp32cu`) and depends on an
  external checkout/build command.
