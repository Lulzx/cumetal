# frontend

Phase 3: Clang CUDA frontend integration lives here.

Current status:

- `cumetalc` now has an initial `.cu` frontend path that shells out to
  `xcrun clang++ -S -emit-llvm` with minimal CUDA-qualifier defines.
- Output LLVM IR is fed into existing AIR emission modes (`experimental`/`xcrun`).
- Coverage exists in `air_abi_cumetalc_cu_experimental_validate`.
