#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__clang__) && defined(__CUDA__)
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
#ifndef __managed__
#define __managed__ __attribute__((managed))
#endif
#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(t, b) __attribute__((launch_bounds(t, b)))
#endif
#endif

#ifndef __align__
#if defined(__clang__) || defined(__GNUC__)
#define __align__(n) __attribute__((aligned(n)))
#else
#define __align__(n)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchTimeout = 6,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorNotReady = 34,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIllegalAddress = 700,
    cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind;

typedef struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
#ifdef __cplusplus
    constexpr uint3(unsigned int vx = 0, unsigned int vy = 0, unsigned int vz = 0)
        : x(vx), y(vy), z(vz) {}
#endif
} uint3;

typedef struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
#ifdef __cplusplus
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3(void) const { return uint3{x, y, z}; }
#endif
} dim3;

typedef struct __align__(16) float4 {
    float x;
    float y;
    float z;
    float w;
} float4;

#ifdef __cplusplus
static inline constexpr float4 make_float4(float x, float y, float z, float w) {
    return float4{x, y, z, w};
}
#endif

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
    int unifiedAddressing;       // Always 1 on Apple Silicon (UMA)
    int managedMemory;           // Always 1 on Apple Silicon (UMA)
    int concurrentManagedAccess; // Always 1 on Apple Silicon (UMA)
    int maxBufferArguments;      // 31 (Metal buffer argument limit)
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

typedef enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3,
} cudaFuncCache;

typedef enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2,
} cudaSharedMemConfig;

typedef struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
} cudaFuncAttributes;

typedef enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
} cudaMemoryType;

typedef struct cudaPointerAttributes {
    cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
} cudaPointerAttributes;

typedef struct cudaStream_st* cudaStream_t;
typedef struct cudaEvent_st* cudaEvent_t;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* user_data);

#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)

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
    cudaHostAllocPortable = 0x1,
    cudaHostAllocMapped = 0x2,
    cudaHostAllocWriteCombined = 0x4,
};

enum {
    cudaDeviceScheduleAuto = 0x00,
    cudaDeviceScheduleSpin = 0x01,
    cudaDeviceScheduleYield = 0x02,
    cudaDeviceScheduleBlockingSync = 0x04,
    cudaDeviceMapHost = 0x08,
    cudaDeviceLmemResizeToMax = 0x10,
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
cudaError_t cudaDriverGetVersion(int* driver_version);
cudaError_t cudaRuntimeGetVersion(int* runtime_version);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int* flags);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device);
cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);
cudaError_t cudaMalloc(void** dev_ptr, size_t size);
cudaError_t cudaMallocManaged(void** dev_ptr, size_t size, unsigned int flags);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* flags, void* host_ptr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaFree(void* dev_ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst,
                            const void* src,
                            size_t count,
                            cudaMemcpyKind kind,
                            cudaStream_t stream);
cudaError_t cudaMemcpyToSymbol(const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbolAsync(const void* symbol,
                                    const void* src,
                                    size_t count,
                                    size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsync(void* dst,
                                      const void* symbol,
                                      size_t count,
                                      size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream);
cudaError_t cudaMemset(void* dev_ptr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
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
cudaError_t cudaConfigureCall(dim3 grid_dim,
                              dim3 block_dim,
#ifdef __cplusplus
                              size_t shared_mem = 0,
                              cudaStream_t stream = nullptr);
#else
                              size_t shared_mem,
                              cudaStream_t stream);
#endif
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void* func);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                          const void* func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize,
                                               int* blockSize,
                                               const void* func,
                                               size_t dynamicSMemSize,
                                               int blockSizeLimit);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr);
cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop);

typedef enum cudaLimit {
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02,
    cudaLimitDevRuntimeSyncDepth = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04,
    cudaLimitMaxL2FetchGranularity = 0x05,
    cudaLimitPersistingL2CacheSize = 0x06,
} cudaLimit;

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit);
cudaError_t cudaLaunchCooperativeKernel(const void* func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMem,
                                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && defined(__clang__) && defined(__CUDA__)

#include <limits.h>
#include <math.h>

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>

#if defined(assert)
#undef assert
#define assert(expr) ((void)0)
#endif

static __host__ __forceinline__ int max(int a, int b) {
    return a > b ? a : b;
}

static __host__ __forceinline__ int min(int a, int b) {
    return a < b ? a : b;
}

template <typename T>
static __device__ __forceinline__ T __ldcs(const T* ptr) {
    return *ptr;
}

// __ldg: load via read-only (texture) cache. On UMA Apple Silicon there is no
// dedicated read-only cache, so this is a plain load — identical semantics,
// no performance difference (spec §8).
template <typename T>
static __device__ __forceinline__ T __ldg(const T* ptr) {
    return *ptr;
}

// __ldlu/__ldcv: non-coherent/volatile load hints — plain loads on UMA.
template <typename T>
static __device__ __forceinline__ T __ldlu(const T* ptr) {
    return *ptr;
}

template <typename T>
static __device__ __forceinline__ T __ldcv(const T* ptr) {
    return *ptr;
}

template <typename T>
static __device__ __forceinline__ void __stcs(T* ptr, T value) {
    *ptr = value;
}

static __device__ __forceinline__ float atomicAdd(float* ptr, float val) {
    return __fAtomicAdd(ptr, val);
}

static __device__ __forceinline__ int atomicAdd(int* ptr, int val) {
    return __iAtomicAdd(ptr, val);
}

static __device__ __forceinline__ unsigned int atomicAdd(unsigned int* ptr, unsigned int val) {
    return __uAtomicAdd(ptr, val);
}

static __device__ __forceinline__ unsigned long long atomicAdd(unsigned long long* ptr,
                                                                unsigned long long val) {
    return __ullAtomicAdd(ptr, val);
}

#endif
