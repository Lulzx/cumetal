# CLAUDE.md — CuMetal Development Guide

You are building CuMetal, a CUDA compiler and runtime for Apple Silicon. The full design
specification is in `cumetal-spec-v0.1.md` in this directory. Read it before making any
architectural decisions.

## Project Identity

CuMetal compiles CUDA C++ source files to Apple's Metal GPU stack. The **primary path** is
source recompilation via `cumetalc` (ahead-of-time). A binary compatibility shim exists but
is opt-in and secondary. Do not optimize for the binary shim path at the expense of the
AOT path.

## Current Phase

**Phase 0.5 — Metallib Validation Harness**

Goal: Generate a `.metallib` file from LLVM IR that Apple's Metal driver accepts. No kernel
execution yet — just container acceptance.

Exit criterion: A `.metallib` generated entirely by CuMetal passes:
1. `air_validate` (MetalLibraryArchive parser) without error
2. `xcrun metal -validate` accepts the file (if available)
3. `MTLDevice.newLibraryWithData:` returns a valid `MTLLibrary` on M-series hardware

After Phase 0.5, move to Phase 0 (kernel execution), then Phases 1–5 per the spec roadmap.

## Repository Structure

```
cumetal/
├── compiler/
│   ├── frontend/          # Clang CUDA → LLVM IR bridge (Phase 3)
│   ├── ptx/               # PTX text parser → LLVM IR (Phase 1)
│   ├── passes/            # LLVM transformation passes (Phase 1)
│   │   ├── intrinsic_lower.cpp
│   │   ├── addrspace.cpp
│   │   ├── printf_lower.cpp
│   │   └── metadata.cpp
│   ├── air_emitter/       # LLVM bitcode → .metallib container (Phase 0.5)
│   ├── air_validate/      # .metallib validation (Phase 0.5)
│   └── cumetalc/          # CLI driver (Phase 3)
├── runtime/
│   ├── api/               # cuda.h, cuda_runtime.h shim headers (Phase 2)
│   ├── driver/            # CUDA Driver API (Phase 2)
│   ├── rt/                # CUDA Runtime API (Phase 2)
│   ├── registration/      # __cudaRegisterFatBinary (Phase 2, opt-in)
│   ├── error/             # Per-thread error queue (Phase 2)
│   ├── metal_backend/     # Obj-C++ Metal wrapper (Phase 0)
│   └── cache/             # .metallib disk cache (Phase 2)
├── tests/
│   ├── unit/
│   ├── functional/
│   ├── conformance/
│   ├── air_abi/
│   └── ptx_sweep/
├── tools/
│   ├── air_inspect/       # .metallib dumper (Phase 0.5)
│   ├── ptx_diff/
│   └── cumetal_bench/
├── install/
├── docs/
├── cumetal-spec-v0.1.md   # THE SPEC — read this first
└── CLAUDE.md              # This file
```

## Key Technical Facts

These are non-negotiable architectural constraints. Do not deviate from them.

- **SIMD width is 32.** Always. On all Apple Silicon. Hardcode it. No runtime queries, no
  emulation modes, no width indirection. See spec §7.
- **UMA means memcpy is memcpy.** `cudaMemcpy` in all directions (H→D, D→H, D→D) is just
  `memcpy()`. No blit encoders for synchronous copies. See spec §6.2.
- **`cudaMallocHost` returns `MTLBuffer.contents`, not `malloc`.** The pointer must be
  trackable by the runtime for kernel argument binding. See spec §6.2.
- **Allocation tracking is mandatory.** A `ptr → MTLBuffer` lookup table must exist from
  Phase 0. Every `cudaMalloc*` registers; every `cudaFree` deregisters; every kernel launch
  resolves pointer args through it. See spec §6.2.1.
- **AIR intrinsic names in the spec are provisional.** They are plausible but must be
  validated against real Metal compiler `.metallib` output in Phase 0.5. When you disassemble
  real metallibs, update the spec's intrinsic tables with the actual names.
- **Kernel arguments are mixed.** Pointers go through `setBuffer:offset:atIndex:`. Scalars
  and small structs (≤ 4 KB) go through `setBytes:length:atIndex:`. Do not assume all args
  are pointers.
- **Constant buffer index 30 is reserved.** `__constant__` memory is collected into a single
  buffer per module, bound at index 30. Printf buffer is at the last argument index.
- **Max buffer arguments is 31.** Metal hard limit. Kernels with many parameters must be
  tested against this.

## Build System

- **Language:** C++ for LLVM passes and runtime, Objective-C++ for Metal backend, Rust for
  PTX parser.
- **Build:** CMake 3.28+. LLVM 18+ is a build dependency.
- **Required:** Xcode Command Line Tools 15+, macOS SDK 14.0+.
- **Optional:** MetalLibraryArchive (Swift package, for `air_validate`).

Build targets:
```
cumetalc              # AOT compiler CLI
libcumetal.dylib      # Runtime + JIT (primary)
libcumetal.a          # Static link option
libcuda.dylib         # Binary shim (only with -DCUMETAL_ENABLE_BINARY_SHIM=ON)
cumetal_tests         # Tests
cumetal_bench         # Benchmarks
```

## Coding Conventions

- **Metal API calls go through `runtime/metal_backend/` only.** No direct `MTL*` calls from
  `runtime/driver/` or `runtime/rt/`. The metal backend is the single Obj-C++ boundary.
- **Error handling:** Every CUDA API function sets the per-thread error via the error queue
  in `runtime/error/`. Never silently swallow errors.
- **Thread safety:** The allocation table, kernel registry, and stream list are accessed from
  multiple threads. Use lock-free structures or fine-grained locks. No global mutexes.
- **No Apple private APIs.** Only public Metal framework headers. The AIR format is reverse-
  engineered from public `.metallib` outputs, not private headers.
- **No NVIDIA headers.** All `cuda.h` / `cuda_runtime.h` content is clean-room. If you need
  to know a CUDA API signature, read the public CUDA documentation, never NVIDIA's headers.
- **Tests for everything.** Every LLVM pass gets FileCheck tests. Every runtime API gets a
  functional test. No untested code paths.

## Phase 0.5 Implementation Plan

This is what to build first:

### Step 1: air_inspect tool
Write a tool that takes a `.metallib` file (compiled by Apple's `metal` compiler from a
trivial Metal kernel) and dumps its structure: magic header, function list, per-function
bitcode, metadata fields. Output format: human-readable text.

How to get a reference `.metallib`:
```metal
// reference.metal
kernel void vector_add(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}
```
```bash
xcrun metal -c reference.metal -o reference.air
xcrun metallib reference.air -o reference.metallib
```

Disassemble the `.metallib` and the `.air` with every tool available. Document findings in
`docs/air-abi.md`.

### Step 2: air_emitter
Write a component that takes LLVM IR with AIR metadata and serializes it into the
`.metallib` container format. The container structure should match what `air_inspect` found
in Step 1.

Start with the simplest possible kernel (vector add). Hand-write the LLVM IR:
```llvm
define void @vector_add(ptr addrspace(1) %a, ptr addrspace(1) %b,
                        ptr addrspace(1) %c, i32 %id) #0 {
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %id
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %id
  %pc = getelementptr float, ptr addrspace(1) %c, i32 %id
  %va = load float, ptr addrspace(1) %pa
  %vb = load float, ptr addrspace(1) %pb
  %vc = fadd float %va, %vb
  store float %vc, ptr addrspace(1) %pc
  ret void
}

; Metadata — exact fields TBD from Step 1 disassembly
attributes #0 = { "air.kernel" "air.version"="2.6" }
!air.kernel = !{!0}
!0 = !{ptr @vector_add, ...}
```

The metadata fields will need to match exactly what Apple's Metal driver expects. This is
the hard part. Use the reference `.metallib` from Step 1 as ground truth.

### Step 3: air_validate
Integrate MetalLibraryArchive (or write a minimal parser) that reads the CuMetal-generated
`.metallib` and verifies:
- Container magic and version fields are correct
- Function list is parseable
- Per-function bitcode is valid LLVM bitcode
- Required metadata fields are present

### Step 4: Metal driver acceptance test
Write a minimal Obj-C++ test program:
```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
NSData* data = [NSData dataWithContentsOfFile:@"cumetal_vector_add.metallib"];
NSError* error = nil;
id<MTLLibrary> lib = [device newLibraryWithData:dispatch_data_create(data.bytes, data.length, nil, nil) error:&error];
// If lib != nil and error == nil: Phase 0.5 exit criterion met
```

## What Not To Do

- **Do not start the PTX parser before Phase 0.5 is done.** If you can't generate a valid
  `.metallib` container, nothing else matters.
- **Do not write the CUDA runtime headers before Phase 2.** Focus on the compiler pipeline
  first.
- **Do not optimize for performance before Phase 4.** Correctness only. A slow correct
  result is infinitely better than a fast wrong one.
- **Do not implement the binary shim (`__cudaRegisterFatBinary`) in Phase 2.** The Phase 2
  exit criterion uses `cumetalc` (source recompilation), not binary drop-in.
- **Do not assume AIR intrinsic names from the spec are correct.** Validate them against
  real Metal compiler output. The spec's tables are educated guesses.
- **Do not use `DYLD_LIBRARY_PATH` anywhere.** Always `DYLD_FALLBACK_LIBRARY_PATH` or
  `@rpath`. See spec §6.9.
- **Do not ship anything that imports NVIDIA headers.** Clean-room only.

## When Stuck

1. **AIR metadata format unclear:** Compile more Metal kernels with different signatures
   (varying arg counts, types, threadgroup memory) and disassemble. The pattern will emerge.
2. **LLVM IR not accepted by Metal driver:** Diff the bitcode of your generated `.metallib`
   against the reference one byte-by-byte. The divergence point reveals the format error.
3. **Intrinsic mapping unknown:** Write the equivalent kernel in Metal Shading Language,
   compile to `.air`, disassemble the bitcode with `llvm-dis`, and read the intrinsic calls.
4. **Runtime behavior unclear:** Write the equivalent program using Metal API directly (no
   CUDA), verify it works, then make the CUDA shim produce the same Metal API call sequence.
5. **Legal question:** If in doubt about clean-room status, do not look at the source in
   question. Write the implementation from the public API documentation only.

## Reference Commands

```bash
# Compile Metal shader to .metallib (reference for reverse engineering)
xcrun metal -c shader.metal -o shader.air
xcrun metallib shader.air -o shader.metallib

# Disassemble AIR bitcode
xcrun metal -S shader.metal -o shader.ll    # to LLVM IR text
llvm-dis shader.air -o shader.air.ll        # if air is raw bitcode

# Inspect metallib container
# (use air_inspect once built, or hexdump for now)
xxd shader.metallib | head -100

# Build CuMetal
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run tests
./build/cumetal_tests

# Run benchmarks (Phase 5+)
./build/cumetal_bench
```

## Success Criteria Per Phase

| Phase | Ship When |
|-------|-----------|
| 0.5 | `.metallib` accepted by `MTLDevice.newLibraryWithData:` |
| 0 | Vector add kernel produces correct output on M1+ |
| 1 | Unmodified PTX for vector add + matmul → correct output |
| 2 | `samples/vectorAdd` compiles with `cumetalc` and runs |
| 3 | Arbitrary `.cu` files compile with `cumetalc` |
| 4 | Conformance suite ≥ 90%; `llm.c` test passes |
| 4.5 | cuBLAS GEMM works via MPS backend |
| 5 | ≤ 2× native Metal for memory-bound kernels |
