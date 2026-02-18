#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cumetal::metal_backend {

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual void* contents() const = 0;
    virtual std::size_t length() const = 0;
};

struct KernelArg {
    enum class Kind {
        kBuffer,
        kBytes,
    };

    Kind kind = Kind::kBytes;
    std::shared_ptr<Buffer> buffer;
    std::size_t offset = 0;
    std::vector<std::uint8_t> bytes;
};

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t shared_memory_bytes = 0;
};

cudaError_t initialize(std::string* error_message);
cudaError_t allocate_buffer(std::size_t size,
                            std::shared_ptr<Buffer>* out_buffer,
                            std::string* error_message);
cudaError_t launch_kernel(const std::string& metallib_path,
                          const std::string& kernel_name,
                          const LaunchConfig& config,
                          const std::vector<KernelArg>& args,
                          std::string* error_message);
cudaError_t synchronize(std::string* error_message);

}  // namespace cumetal::metal_backend
