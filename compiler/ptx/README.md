# ptx

Phase 1: PTX parsing and LLVM IR lowering.

Current status:

- `cumetal::ptx::parse_ptx` parses:
  - `.version`
  - `.target`
  - `.entry` signatures
  - `.param` declarations in entry parameter lists
- This is an initial text parser scaffold for upcoming lowering work.
