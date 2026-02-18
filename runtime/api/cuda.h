#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUdevice;
typedef uint64_t CUdeviceptr;

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct cudaStream_st* CUstream;
typedef struct cudaEvent_st* CUevent;

enum {
    CU_STREAM_DEFAULT = 0x0,
    CU_STREAM_NON_BLOCKING = 0x1,
};

typedef enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_UNKNOWN = 999,
} CUresult;

CUresult cuInit(unsigned int flags);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSynchronize(void);

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamQuery(CUstream hStream);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags);

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventQuery(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);

CUresult cuModuleLoad(CUmodule* module, const char* fname);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra);

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);

CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuGetErrorString(CUresult error, const char** pStr);

#ifdef __cplusplus
}
#endif
