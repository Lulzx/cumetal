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
- `printf_lower` scaffold maps PTX `call*` sites targeting `printf`/`vprintf`:
  - emits deduplicated format-table entries (256-byte literal cap)
  - captures per-call format id + argument list for runtime buffer encoding
  - strict mode rejects malformed calls (e.g., missing argument tuple)
- `metadata` scaffold emits kernel metadata fields:
  - `air.kernel`, `air.version`, `language.version`
  - per-argument `type` and `name` records
  - `kernel.printf.*` format-table records from `printf_lower`
- `phase1_pipeline` scaffold chains parser + passes for one PTX entry and returns
  lowered instructions, printf-lowered calls, addrspace rewrites, metadata, and warnings.
- This pipeline is now consumed by `cumetal::ptx::lower_ptx_to_llvm_ir`.
