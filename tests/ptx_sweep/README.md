# ptx_sweep

Per-opcode PTX sweep scaffolding for Phase 1.

- `run_supported_ops.sh`: emits minimal PTX kernels for currently mapped opcodes and
  requires strict PTX->LLVM lowering success.
  - arithmetic: `add`, `sub`, `mul`, `mad`
  - special-register moves: `%tid.*`, `%ctaid.*`, `%ntid.*`, `%nctaid.*`
  - memory/addrspace: `ld/st` in shared/global/local + `cvta.to.*`
  - control/other: `bar.sync`, `setp`, `bra`, `call` (`vprintf`)
- `run_unsupported_ops.sh`: verifies strict mode rejects unsupported instruction roots
  (`foo`, `trap`, `tex`, `suld`).
