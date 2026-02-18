# ptx

Phase 1: PTX parsing and LLVM IR lowering.

Current status:

- `cumetal::ptx::parse_ptx` parses:
  - `.version`
  - `.target`
  - `.entry` signatures
  - `.param` declarations in entry parameter lists
  - instruction streams per entry (`opcode`, operands, optional predicate)
- Unsupported opcodes are recorded as warnings by default.
- Strict mode (`ParseOptions.strict=true`) fails parsing on unsupported opcodes.
- `cumetal::ptx::lower_ptx_to_llvm_ir` lowers PTX through the phase1 pipeline and emits
  LLVM IR text with AIR-style kernel metadata.
  - For `vector_add`-shaped signatures, a concrete `fadd`/`load`/`store` kernel body is emitted.
- `cumetal-ptx2llvm` CLI writes `.ll` from `.ptx`:
  - `cumetal-ptx2llvm --input kernel.ptx --output kernel.ll --entry kernel_name`
- `tests/ptx_sweep/` now includes initial strict-mode instruction sweep coverage.
