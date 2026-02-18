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

class Stream {
public:
    virtual ~Stream() = default;
};

struct DeviceProperties {
    std::string name;
    std::size_t total_global_mem = 0;
    int shared_mem_per_block = 0;
    int max_threads_per_block = 0;
    int multi_processor_count = 8;
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
cudaError_t query_device_properties(DeviceProperties* out_properties, std::string* error_message);
cudaError_t allocate_buffer(std::size_t size,
                            std::shared_ptr<Buffer>* out_buffer,
                            std::string* error_message);
cudaError_t create_stream(std::shared_ptr<Stream>* out_stream, std::string* error_message);
cudaError_t destroy_stream(const std::shared_ptr<Stream>& stream, std::string* error_message);
cudaError_t stream_synchronize(const std::shared_ptr<Stream>& stream, std::string* error_message);
cudaError_t stream_tail_ticket(const std::shared_ptr<Stream>& stream,
                               std::uint64_t* out_ticket,
                               std::string* error_message);
cudaError_t stream_query_ticket(const std::shared_ptr<Stream>& stream,
                                std::uint64_t ticket,
                                bool* out_complete,
                                std::string* error_message);
cudaError_t stream_wait_ticket(const std::shared_ptr<Stream>& stream,
                               std::uint64_t ticket,
                               std::string* error_message);
cudaError_t launch_kernel(const std::string& metallib_path,
                          const std::string& kernel_name,
                          const LaunchConfig& config,
                          const std::vector<KernelArg>& args,
                          const std::shared_ptr<Stream>& stream,
                          std::string* error_message);
cudaError_t synchronize(std::string* error_message);

}  // namespace cumetal::metal_backend
