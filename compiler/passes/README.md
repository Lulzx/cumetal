# passes

Phase 1: LLVM transformation passes (`intrinsic_lower`, `addrspace`, `printf_lower`, `metadata`).

Current status:

- `intrinsic_lower` scaffold exists and maps parsed PTX instructions for:
  - thread-index special registers (`%tid.*`, `%ctaid.*`, `%ntid.*`, `%nctaid.*`)
  - barriers (`bar.sync*`)
  - basic math (`add*`, `sub*`, `mul*`, `mad*`)
- strict mode reports unmapped opcodes as hard errors.
- `addrspace` rewrite scaffold maps:
  - `ld.*`/`st.*` in `.shared`, `.global`, `.local` spaces
  - `cvta.to.*` casts to explicit LLVM address spaces
- `metadata` scaffold emits kernel metadata fields:
  - `air.kernel`, `air.version`, `language.version`
  - per-argument `type` and `name` records
