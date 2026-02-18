# MetalLibraryArchive Bridge

This optional bridge lets `air_validate` delegate function-list and metadata parsing to
[MetalLibraryArchive](https://github.com/YuAo/MetalLibraryArchive).

## Build

```bash
cd tools/metal_library_archive_bridge
swift build -c release
```

## Run directly

```bash
swift run --package-path tools/metal_library_archive_bridge cumetal-mla-validate path/to/file.metallib
```

## Use with air_validate

```bash
./build/air_validate path/to/file.metallib --mla
# or custom command
./build/air_validate path/to/file.metallib --mla --mla-cmd "swift run --package-path tools/metal_library_archive_bridge cumetal-mla-validate"
```
