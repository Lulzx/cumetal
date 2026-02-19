# CuMetal Intrinsic Map

Complete CUDA/PTX → AIR intrinsic mapping table for the CuMetal intrinsic lowering pass.

---

## Thread Indexing

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `mov.u32 %r, %tid.x` | `air.thread_position_in_threadgroup.x` | Thread index within threadgroup |
| `mov.u32 %r, %tid.y` | `air.thread_position_in_threadgroup.y` | |
| `mov.u32 %r, %tid.z` | `air.thread_position_in_threadgroup.z` | |
| `mov.u32 %r, %ctaid.x` | `air.threadgroup_position_in_grid.x` | Block/threadgroup index |
| `mov.u32 %r, %ctaid.y` | `air.threadgroup_position_in_grid.y` | |
| `mov.u32 %r, %ctaid.z` | `air.threadgroup_position_in_grid.z` | |
| `mov.u32 %r, %ntid.x` | `air.threads_per_threadgroup.x` | Block dimension |
| `mov.u32 %r, %ntid.y` | `air.threads_per_threadgroup.y` | |
| `mov.u32 %r, %ntid.z` | `air.threads_per_threadgroup.z` | |
| `mov.u32 %r, %nctaid.x` | `air.threadgroups_per_grid.x` | Grid dimension |
| `mov.u32 %r, %nctaid.y` | `air.threadgroups_per_grid.y` | |
| `mov.u32 %r, %nctaid.z` | `air.threadgroups_per_grid.z` | |

---

## Synchronization

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `bar.sync N` | `air.threadgroup_barrier` | `__syncthreads()` — threadgroup scope |
| `bar.warp.sync mask` | `air.simdgroup.barrier` | `__syncwarp(mask)` — simdgroup scope; mask emulation: non-0xFFFFFFFF masks conservatively emit full-group barrier |
| `membar.gl` / `__threadfence()` | `air.mem.barrier(scope=device)` | Device-wide memory fence |
| `membar.cta` / `__threadfence_block()` | `air.mem.barrier(scope=threadgroup)` | Threadgroup memory fence |

---

## Warp / SIMD-group Primitives

Apple Silicon SIMD-group width is architecturally fixed at 32 (matching CUDA warp size).

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `shfl.sync.idx.b32 dst, src, lane, clamp, mask` | `air.simdgroup.shuffle` | `__shfl_sync` — index shuffle |
| `shfl.sync.down.b32 dst, src, delta, clamp, mask` | `air.simdgroup.shuffle_down` | `__shfl_down_sync` |
| `shfl.sync.up.b32 dst, src, delta, clamp, mask` | `air.simdgroup.shuffle_up` | `__shfl_up_sync` |
| `shfl.sync.bfly.b32 dst, src, lanexor, clamp, mask` | `air.simdgroup.shuffle_xor` | `__shfl_xor_sync` |
| `vote.sync.ballot.b32 dst, pred, mask` | `air.simdgroup.ballot` | `__ballot_sync` — 32-bit lane mask |
| `vote.sync.any.pred dst, pred, mask` | `air.simdgroup.any` | `__any_sync` |
| `vote.sync.all.pred dst, pred, mask` | `air.simdgroup.all` | `__all_sync` |
| `vote.ballot.b32 dst, pred` | `air.simdgroup.ballot` | Non-sync form |
| `vote.any.pred dst, pred` | `air.simdgroup.any` | Non-sync form |
| `vote.all.pred dst, pred` | `air.simdgroup.all` | Non-sync form |

**Mask semantics note**: AIR simdgroup operations are implicitly full-group. When `mask != 0xFFFFFFFF`, lanes not in the mask read their own value (identity) rather than being truly inactive. This is a conservative, safe divergence from CUDA semantics. Kernels using partial masks should be tested carefully.

---

## Arithmetic and Logic

| PTX Opcode Root | LLVM IR Intrinsic | Notes |
|----------------|-------------------|-------|
| `add` | `llvm.add` | Integer add; `add.f32` → float add |
| `sub` | `llvm.sub` | |
| `mul` | `llvm.mul` | |
| `div` | `llvm.div` | |
| `rem` | `llvm.rem` | |
| `and` | `llvm.and` | |
| `or` | `llvm.or` | |
| `xor` | `llvm.xor` | |
| `not` | `llvm.not` | |
| `shl` | `llvm.shl` | |
| `shr` | `llvm.shr` | |
| `neg.f32` | `llvm.fneg` | Float negate |
| `neg.s32` | `llvm.neg` | Integer negate |
| `selp` | `llvm.select` | Conditional select |
| `rcp` | `llvm.rcp` | Reciprocal |
| `mad.f32` | `llvm.fma` | Fused multiply-add (float) |
| `mad.s32` | `llvm.mad` | Multiply-add (integer) |
| `fma` | `llvm.fma` | Fused multiply-add |
| `max.f32` | `llvm.fmax` | Float max |
| `max.s32` | `llvm.max` | Integer max |
| `min.f32` | `llvm.fmin` | Float min |
| `min.s32` | `llvm.min` | Integer min |
| `abs.f32` | `llvm.fabs` | Float absolute value |
| `abs.s32` | `llvm.abs` | Integer absolute value |

---

## Math Intrinsics

| PTX Opcode | LLVM IR / AIR Intrinsic | Notes |
|------------|------------------------|-------|
| `sqrt.rn.f32` | `llvm.sqrt` | Square root |
| `rsqrt.approx.f32` | `llvm.rsqrt` | Reciprocal square root |
| `ex2.approx.f32` | `llvm.exp2` | 2^x (exp base 2) |
| `lg2.approx.f32` | `llvm.log2` | log₂(x) |
| `sin.approx.f32` | `llvm.sin` | Sine; precision may differ ≤1 ULP from NVIDIA |
| `cos.approx.f32` | `llvm.cos` | Cosine; precision may differ ≤1 ULP from NVIDIA |

---

## Memory

| PTX Opcode | Lowering | Notes |
|------------|----------|-------|
| `ld.param.*` | Param load — handled structurally | Mapped to kernel argument index |
| `ld.global.f32 dst, [addr]` | `device float* ptr; dst = ptr[gid]` | Direct Metal buffer access |
| `st.global.f32 [addr], src` | `device float* ptr; ptr[gid] = src` | Direct Metal buffer store |
| `atom.global.add.f32 dst, [addr], src` | `atomic_fetch_add_explicit(...)` | Global atomic add |
| `atom.global.add.f64` | `llvm.atomic.add.f64` | FP64 atomics via LLVM IR |

---

## Passthrough (no lowering needed)

The following PTX opcodes are passed through to subsequent pipeline stages unchanged:

`ret`, `ld` (non-global), `st` (non-global), `setp`, `bra`, `cvt`, `cvta`, `mov` (non-special-register), `call`, `atom`

---

## Unsupported / Compile-time Errors

| PTX Feature | Status |
|-------------|--------|
| `cluster.*`, `mbarrier.*` | Per-instruction compile-time error |
| `cp.async.bulk.tensor.*` (TMA) | Per-instruction compile-time error |
| `cvt.rn.f8x2.*` (FP8) | Per-instruction compile-time error |
| Dynamic parallelism (`launch_cooperative_kernel`) | Compile-time error |
