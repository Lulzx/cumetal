# cumetalc

`cumetalc` is the command-line compiler driver entrypoint.

Current implementation is a thin wrapper over `cumetal-air-emitter`:

```bash
cumetalc --mode xcrun --input kernel.metal --output kernel.metallib --overwrite
```

Future phases extend this to CUDA source (`.cu`) compilation.
