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
| `mov.u32 %r, %laneid` | `air.thread_position_in_simdgroup` | Lane index within warp/SIMD-group (0–31) |
| `mov.u32 %r, %warpsize` | `air.constant.warp_size` | Architecturally fixed at 32 (spec §7) |
| `mov.u32 %r, %lanemask_eq` | `air.simdgroup.lanemask_eq` | Bitmask: only current lane bit set |
| `mov.u32 %r, %lanemask_lt` | `air.simdgroup.lanemask_lt` | Bitmask: lanes with index < laneid |
| `mov.u32 %r, %lanemask_le` | `air.simdgroup.lanemask_le` | Bitmask: lanes with index ≤ laneid |
| `mov.u32 %r, %lanemask_gt` | `air.simdgroup.lanemask_gt` | Bitmask: lanes with index > laneid |
| `mov.u32 %r, %lanemask_ge` | `air.simdgroup.lanemask_ge` | Bitmask: lanes with index ≥ laneid |

---

## Synchronization

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `bar.sync N` | `air.threadgroup_barrier` | `__syncthreads()` — threadgroup scope |
| `bar.warp.sync mask` | `air.simdgroup.barrier` | `__syncwarp(mask)` — simdgroup scope; non-0xFFFFFFFF masks conservatively emit full-group barrier |
| `membar.gl` | `air.mem.barrier.device` | `__threadfence()` — device-wide memory fence |
| `membar.sys` | `air.mem.barrier.device` | `__threadfence_system()` — system-wide fence (lowered to device scope) |
| `membar.cta` | `air.mem.barrier.threadgroup` | `__threadfence_block()` — threadgroup memory fence |

## Async Copy

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `cp.async.ca.shared.global [dst], [src], sz` | `air.cp_async` | Async copy global→shared; lowered to synchronous ld+st (functional, not performance-equivalent) |
| `cp.async.commit_group` | `air.threadgroup_barrier` | Commit outstanding async copies (barrier serializes) |
| `cp.async.wait_group N` | `air.threadgroup_barrier` | Wait until ≤N outstanding groups remain (barrier serializes) |
| `cp.async.wait_all` | `air.threadgroup_barrier` | Wait for all outstanding async copies |

---

## Warp / SIMD-group Reductions

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `redux.sync.add.s32 dst, src, mask` | `air.simdgroup.reduce_add` | `__redux_sync` warp-wide add (integer) |
| `redux.sync.add.f32 dst, src, mask` | `air.simdgroup.reduce_add.f32` | Warp-wide add (float) |
| `redux.sync.and.b32 dst, src, mask` | `air.simdgroup.reduce_and` | Warp-wide bitwise AND |
| `redux.sync.or.b32 dst, src, mask` | `air.simdgroup.reduce_or` | Warp-wide bitwise OR |
| `redux.sync.xor.b32 dst, src, mask` | `air.simdgroup.reduce_xor` | Warp-wide bitwise XOR |
| `redux.sync.min.s32 dst, src, mask` | `air.simdgroup.reduce_min` | Warp-wide min (integer) |
| `redux.sync.min.f32 dst, src, mask` | `air.simdgroup.reduce_min.f32` | Warp-wide min (float) |
| `redux.sync.max.s32 dst, src, mask` | `air.simdgroup.reduce_max` | Warp-wide max (integer) |
| `redux.sync.max.f32 dst, src, mask` | `air.simdgroup.reduce_max.f32` | Warp-wide max (float) |

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

| `clz.b32` | `llvm.ctlz.i32` | Count leading zeros (32-bit) |
| `clz.b64` | `llvm.ctlz.i64` | Count leading zeros (64-bit) |
| `popc.b32` | `llvm.ctpop.i32` | Population count (32-bit) |
| `popc.b64` | `llvm.ctpop.i64` | Population count (64-bit) |
| `brev.b32` | `llvm.bitreverse.i32` | Bit reverse (32-bit) |
| `brev.b64` | `llvm.bitreverse.i64` | Bit reverse (64-bit) |

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
