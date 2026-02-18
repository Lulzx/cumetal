#include "metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace cumetal::metal_backend {
namespace {

class BufferImpl final : public Buffer {
public:
    explicit BufferImpl(id<MTLBuffer> buffer) : buffer_(buffer) {}

    void* contents() const override {
        return [buffer_ contents];
    }

    std::size_t length() const override {
        return static_cast<std::size_t>([buffer_ length]);
    }

    id<MTLBuffer> handle() const {
        return buffer_;
    }

private:
    id<MTLBuffer> buffer_;
};

struct BackendState {
    std::mutex mutex;
    bool initialized = false;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unordered_map<std::string, id<MTLLibrary>> library_cache;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
};

BackendState& state() {
    static BackendState kState;
    return kState;
}

bool ensure_initialized(std::string* error_message) {
    BackendState& backend = state();
    std::lock_guard<std::mutex> lock(backend.mutex);
    if (backend.initialized) {
        return true;
    }

    @autoreleasepool {
        backend.device = MTLCreateSystemDefaultDevice();
        if (backend.device == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create default Metal device";
            }
            return false;
        }

        backend.queue = [backend.device newCommandQueue];
        if (backend.queue == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create Metal command queue";
            }
            return false;
        }

        backend.initialized = true;
        return true;
    }
}

id<MTLLibrary> load_library_locked(BackendState& backend,
                                   const std::string& metallib_path,
                                   std::string* error_message) {
    const auto found = backend.library_cache.find(metallib_path);
    if (found != backend.library_cache.end()) {
        return found->second;
    }

    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
        NSError* read_error = nil;
        NSData* data = [NSData dataWithContentsOfFile:path options:0 error:&read_error];
        if (data == nil || [data length] == 0) {
            if (error_message != nullptr) {
                *error_message = "failed to read metallib file: " + metallib_path;
                if (read_error != nil) {
                    *error_message += " (" + std::string([[read_error localizedDescription] UTF8String]) + ")";
                }
            }
            return nil;
        }

        dispatch_data_t dispatch_data = dispatch_data_create(
            [data bytes], [data length], dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        NSError* library_error = nil;
        id<MTLLibrary> library = [backend.device newLibraryWithData:dispatch_data error:&library_error];
        if (library == nil) {
            if (error_message != nullptr) {
                *error_message = "newLibraryWithData failed for metallib: " + metallib_path;
                if (library_error != nil) {
                    *error_message +=
                        " (" + std::string([[library_error localizedDescription] UTF8String]) + ")";
                }
            }
            return nil;
        }

        backend.library_cache.emplace(metallib_path, library);
        return library;
    }
}

id<MTLComputePipelineState> load_pipeline_locked(BackendState& backend,
                                                 const std::string& metallib_path,
                                                 const std::string& kernel_name,
                                                 std::string* error_message) {
    const std::string cache_key = metallib_path + "::" + kernel_name;
    const auto found = backend.pipeline_cache.find(cache_key);
    if (found != backend.pipeline_cache.end()) {
        return found->second;
    }

    id<MTLLibrary> library = load_library_locked(backend, metallib_path, error_message);
    if (library == nil) {
        return nil;
    }

    @autoreleasepool {
        NSString* function_name = [NSString stringWithUTF8String:kernel_name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to find kernel function: " + kernel_name;
            }
            return nil;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [backend.device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create compute pipeline for function: " + kernel_name;
                if (pipeline_error != nil) {
                    *error_message +=
                        " (" + std::string([[pipeline_error localizedDescription] UTF8String]) + ")";
                }
            }
            return nil;
        }

        backend.pipeline_cache.emplace(cache_key, pipeline);
        return pipeline;
    }
}

}  // namespace

cudaError_t initialize(std::string* error_message) {
    return ensure_initialized(error_message) ? cudaSuccess : cudaErrorInitializationError;
}

cudaError_t allocate_buffer(std::size_t size,
                            std::shared_ptr<Buffer>* out_buffer,
                            std::string* error_message) {
    if (out_buffer == nullptr || size == 0) {
        if (error_message != nullptr) {
            *error_message = "allocate_buffer invalid argument";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    BackendState& backend = state();
    std::lock_guard<std::mutex> lock(backend.mutex);

    id<MTLBuffer> buffer = [backend.device newBufferWithLength:size options:MTLStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "newBufferWithLength failed";
        }
        return cudaErrorMemoryAllocation;
    }

    *out_buffer = std::make_shared<BufferImpl>(buffer);
    return cudaSuccess;
}

cudaError_t launch_kernel(const std::string& metallib_path,
                          const std::string& kernel_name,
                          const LaunchConfig& config,
                          const std::vector<KernelArg>& args,
                          std::string* error_message) {
    if (metallib_path.empty() || kernel_name.empty()) {
        if (error_message != nullptr) {
            *error_message = "launch_kernel requires metallib path and kernel name";
        }
        return cudaErrorInvalidValue;
    }

    if (config.grid.x == 0 || config.grid.y == 0 || config.grid.z == 0 || config.block.x == 0 ||
        config.block.y == 0 || config.block.z == 0) {
        if (error_message != nullptr) {
            *error_message = "launch dimensions must be non-zero";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    BackendState& backend = state();
    std::lock_guard<std::mutex> lock(backend.mutex);

    id<MTLComputePipelineState> pipeline =
        load_pipeline_locked(backend, metallib_path, kernel_name, error_message);
    if (pipeline == nil) {
        return cudaErrorInvalidValue;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [backend.queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create command buffer";
            }
            return cudaErrorUnknown;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create compute encoder";
            }
            return cudaErrorUnknown;
        }

        [encoder setComputePipelineState:pipeline];

        for (std::size_t i = 0; i < args.size(); ++i) {
            const KernelArg& arg = args[i];
            if (arg.kind == KernelArg::Kind::kBuffer) {
                if (arg.buffer == nullptr) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) + " missing buffer";
                    }
                    return cudaErrorInvalidValue;
                }

                auto* buffer_impl = dynamic_cast<BufferImpl*>(arg.buffer.get());
                if (buffer_impl == nullptr) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) + " has unexpected buffer type";
                    }
                    return cudaErrorInvalidValue;
                }

                [encoder setBuffer:buffer_impl->handle() offset:arg.offset atIndex:i];
            } else {
                if (arg.bytes.empty()) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) + " has empty byte payload";
                    }
                    return cudaErrorInvalidValue;
                }
                if (arg.bytes.size() > 4096) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) +
                                         " byte payload exceeds 4KB setBytes limit";
                    }
                    return cudaErrorInvalidValue;
                }

                [encoder setBytes:arg.bytes.data() length:arg.bytes.size() atIndex:i];
            }
        }

        if (config.shared_memory_bytes > 0) {
            [encoder setThreadgroupMemoryLength:config.shared_memory_bytes atIndex:0];
        }

        const MTLSize threadgroups = MTLSizeMake(config.grid.x, config.grid.y, config.grid.z);
        const MTLSize threads_per_threadgroup =
            MTLSizeMake(config.block.x, config.block.y, config.block.z);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];

        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* command_error = [command_buffer error];
            if (error_message != nullptr) {
                *error_message = "command buffer failed";
                if (command_error != nil) {
                    *error_message +=
                        " (" + std::string([[command_error localizedDescription] UTF8String]) + ")";
                }
            }
            return cudaErrorUnknown;
        }
    }

    return cudaSuccess;
}

cudaError_t synchronize(std::string* error_message) {
    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }
    return cudaSuccess;
}

}  // namespace cumetal::metal_backend
