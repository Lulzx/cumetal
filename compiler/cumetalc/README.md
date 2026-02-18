# cumetalc

`cumetalc` is the command-line compiler driver entrypoint.

Current implementation is a thin wrapper over `cumetal-air-emitter`:

```bash
cumetalc --mode xcrun --input kernel.metal --output kernel.metallib --overwrite
cumetalc --mode xcrun kernel.metal -o kernel.metallib --overwrite
cumetalc --mode experimental --input kernel.ptx --output kernel.metallib --entry kernel_name --ptx-strict --overwrite
cumetalc --mode experimental --input kernel.cu --output kernel.metallib --overwrite
```

If `-o/--output` is omitted, output defaults to `<input-stem>.metallib`
(for `.metal`, `.ptx`, and `.cu` inputs).

For `.ptx` input, `cumetalc` lowers PTX to temporary LLVM IR internally (via phase1
pipeline) and then invokes the AIR emitter.

For `.cu` input, `cumetalc` invokes `xcrun clang++` with a minimal CUDA-qualifier define
set (`__global__`, `__host__`, `__device__`, `__shared__`, `__constant__`, `__managed__`)
to produce temporary LLVM IR, then runs the AIR emitter.
