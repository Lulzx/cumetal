# cumetalc

`cumetalc` is the command-line compiler driver entrypoint.

Current implementation is a thin wrapper over `cumetal-air-emitter`:

```bash
cumetalc --mode xcrun --input kernel.metal --output kernel.metallib --overwrite
cumetalc --mode xcrun kernel.metal -o kernel.metallib --overwrite
```

If `-o/--output` is omitted, output defaults to `<input-stem>.metallib`.

Future phases extend this to CUDA source (`.cu`) compilation.
