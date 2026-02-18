vectorAdd Sample
================

This sample demonstrates the source-first CuMetal flow:

1. Compile the CUDA kernel with `cumetalc`:

```bash
./build/cumetalc --mode xcrun \
  --input samples/vectorAdd/vectorAdd.cu \
  --output /tmp/vectorAdd.metallib \
  --overwrite
```

2. Compile and link the host app against `libcumetal`:

```bash
xcrun clang++ -std=c++20 \
  samples/vectorAdd/vectorAdd.cpp \
  -Iruntime/api \
  -Lbuild \
  -Wl,-rpath,build \
  -lcumetal \
  -o /tmp/vectorAdd
```

3. Run:

```bash
/tmp/vectorAdd /tmp/vectorAdd.metallib
```
