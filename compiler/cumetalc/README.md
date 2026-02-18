# cumetalc

`cumetalc` is the command-line compiler driver entrypoint.

Current implementation is a thin wrapper over `cumetal-air-emitter`:

```bash
cumetalc --mode xcrun --input kernel.metal --output kernel.metallib --overwrite
cumetalc --mode xcrun kernel.metal -o kernel.metallib --overwrite
cumetalc --mode experimental --input kernel.ptx --output kernel.metallib --entry kernel_name --ptx-strict --overwrite
```

If `-o/--output` is omitted, output defaults to `<input-stem>.metallib`.

For `.ptx` input, `cumetalc` lowers PTX to temporary LLVM IR internally (via phase1
pipeline) and then invokes the AIR emitter.

Future phases extend this to CUDA source (`.cu`) compilation.
