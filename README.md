CuMetal
=======

CuMetal is an experimental CUDA compiler/runtime for Apple Silicon GPUs.

Target:
- macOS 14+
- Apple M-series GPU

Execution model:
- Primary: source recompilation with `cumetalc`
- Secondary: optional binary shim compatibility (`CUMETAL_ENABLE_BINARY_SHIM=ON`)

Quick start
-----------

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Install:

```bash
cmake --install build --prefix /tmp/cumetal-install
```

Main tools:
- `cumetalc`
- `cumetal-air-emitter`
- `air_validate`
- `air_inspect`
- `cumetal-ptx2llvm`
- `scripts/build_llmc_test_gpt2fp32cu.sh` (llm.c CUDA stress build shim, with configurable gradient tolerance patching for CuMetal conformance)

Conformance snapshot
--------------------

- `conformance_llmc_gpt2fp32cu` passes with:
  - `OK (LOGITS)`
  - `LOSS OK`
  - `TENSOR OK`
  - `overall okay: 1`

Documentation
-------------

- Current implementation status and API coverage: [docs/status.md](./docs/status.md)
- Build and validation workflows: [docs/build.md](./docs/build.md)
- Test and conformance workflows: [docs/testing.md](./docs/testing.md)
- Known feature gaps: [docs/known-gaps.md](./docs/known-gaps.md)
- AIR/metallib ABI notes: [docs/air-abi.md](./docs/air-abi.md)
- Design specification: [spec.md](./spec.md)

License
-------

[Apache 2.0](./LICENSE)
