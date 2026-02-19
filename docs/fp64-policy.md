# FP64 Policy

CuMetal's handling of double-precision floating-point (`double`, `f64`) on Apple Silicon.

---

## Hardware Reality

Apple Silicon GPUs have limited FP64 support. The GPU ALUs can execute some FP64 instructions
but at drastically reduced throughput — approximately 1/32 of FP32 throughput. This is
consistent across M1, M2, M3, and M4 families.

This is not a CuMetal limitation; it reflects the hardware. NVIDIA A100 GPUs provide
1:2 FP64:FP32 throughput. Apple Silicon GPUs do not have a high-throughput FP64 path.

---

## Compilation Modes

CuMetal provides three FP64 compilation modes, selected via the `--fp64` flag:

| Mode | Flag | Behavior | Precision |
|------|------|----------|-----------|
| **Native** (default) | `--fp64=native` | Emit AIR FP64 instructions directly | IEEE 754 double (64-bit) |
| **Emulate** | `--fp64=emulate` | Decompose to FP32 pairs via Dekker's algorithm | ~44 bits mantissa |
| **Warn** | `--fp64=warn` | Same as native, but emit a per-instruction compile warning | IEEE 754 double (64-bit) |

---

## Usage Guidance

**`--fp64=native` (default)**

Use this when:
- Your CUDA kernel uses `double` for occasional accumulation or reduction
- FP64 operations are less than ~5% of total GPU instructions
- IEEE 754 double precision is required

The throughput penalty is per FP64 instruction. For kernels where doubles are rare,
the overall impact on wall-clock time is tolerable.

**`--fp64=emulate`**

Use this when:
- ~44-bit mantissa precision is acceptable for your application
- FP64 throughput matters more than full IEEE precision
- Example use cases: some ML training loops, iterative solvers with loose tolerance

Emulation uses Dekker's algorithm to simulate double precision via pairs of `float` values.
This has roughly 4× overhead per FP64 operation vs. native FP32.

**`--fp64=warn`**

Use this when:
- You want to audit FP64 usage in a codebase
- You're unsure how much FP64 is in a kernel
- Debugging unexpected slowness that may be FP64-related

---

## Programs Dominated by FP64

Scientific simulation, CFD, climate modeling, and similar workloads that use `double`
throughout are **out of scope for GPU execution on Apple Silicon**. The hardware simply
does not provide competitive FP64 throughput.

**Recommended alternative**: Apple's AMX (Accelerate Matrix eXtensions) coprocessor
provides full-speed FP64 SIMD on the CPU. Workloads requiring competitive FP64 throughput
on Apple hardware should target AMX via the Accelerate framework or BLAS routines. A
Metal↔AMX bridge is out of scope for CuMetal but trivial to build on top of this runtime.

---

## Precision Notes

| Operation | Native Mode | Emulate Mode |
|-----------|------------|--------------|
| Addition, subtraction | IEEE 754 double | ~44 bits (Dekker) |
| Multiplication | IEEE 754 double | ~44 bits (Dekker) |
| `sqrt.rn.f64` | IEEE 754 double | ~44 bits |
| `fma.rn.f64` | IEEE 754 double | ~44 bits |
| `ex2.approx.f64` | Same as Metal `exp2` (double) | ~44 bits |
| Transcendentals (`sin`, `cos`, `lg2`) | May differ from NVIDIA by ≤1 ULP | ~44 bits |

---

## Known FP64 Gaps

- `atom.global.add.f64` (Ampere atomic): supported via LLVM atomic IR; requires AIR
  backend to emit the correct instruction. Verified on M1/M2.
- `fma.rn.f64`: translated via `llvm.fma.f64`; precision depends on Apple GPU FP64
  hardware (see §13 open question #2 in spec.md).
- No FP64 shuffle primitives: `shfl.sync.idx.f64` would require two 32-bit shuffles.
  Emitted as two `shfl.b32` operations on the high and low halves.
