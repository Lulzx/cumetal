#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorNotReady = 34,
    cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind;

typedef struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
#ifdef __cplusplus
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
#endif
} dim3;

typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int warpSize;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int sharedMemPerBlock;
    int regsPerBlock;
    int major;
    int minor;
} cudaDeviceProp;

typedef enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrConcurrentManagedAccess = 89,
} cudaDeviceAttr;

typedef struct cudaStream_st* cudaStream_t;
typedef struct cudaEvent_st* cudaEvent_t;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* user_data);

enum {
    cudaEventDefault = 0x0,
    cudaEventBlockingSync = 0x1,
    cudaEventDisableTiming = 0x2,
    cudaEventInterprocess = 0x4,
};

enum {
    cudaStreamDefault = 0x0,
    cudaStreamNonBlocking = 0x1,
};

enum {
    cudaHostAllocDefault = 0x0,
};

typedef enum cumetalArgKind {
    CUMETAL_ARG_BUFFER = 0,
    CUMETAL_ARG_BYTES = 1,
} cumetalArgKind_t;

typedef struct cumetalKernelArgInfo {
    cumetalArgKind_t kind;
    uint32_t size_bytes;
} cumetalKernelArgInfo_t;

typedef struct cumetalKernel {
    const char* metallib_path;
    const char* kernel_name;
    uint32_t arg_count;
    const cumetalKernelArgInfo_t* arg_info;
} cumetalKernel_t;

cudaError_t cudaInit(unsigned int flags);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device);
cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);
cudaError_t cudaMalloc(void** dev_ptr, size_t size);
cudaError_t cudaMallocManaged(void** dev_ptr, size_t size, unsigned int flags);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaFree(void* dev_ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst,
                            const void* src,
                            size_t count,
                            cudaMemcpyKind kind,
                            cudaStream_t stream);
cudaError_t cudaMemset(void* dev_ptr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* user_data,
                                  unsigned int flags);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaLaunchKernel(const void* func,
                             dim3 grid_dim,
                             dim3 block_dim,
                             void** args,
                             size_t shared_mem,
                             cudaStream_t stream);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif
