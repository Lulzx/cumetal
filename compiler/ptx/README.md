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
