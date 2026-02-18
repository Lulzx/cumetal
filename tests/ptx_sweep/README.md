# ptx_sweep

Per-opcode PTX sweep scaffolding for Phase 1.

- `run_supported_ops.sh`: emits minimal PTX kernels for currently mapped opcodes and
  requires strict PTX->LLVM lowering success.
- `run_unsupported_ops.sh`: verifies strict mode rejects an unknown opcode.
