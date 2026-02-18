#pragma once

#include <stddef.h>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasContext* cublasHandle_t;

typedef enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
} cublasStatus_t;

typedef enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
} cublasOperation_t;

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream_id);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id);

cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           const float* x,
                           int incx,
                           float* y,
                           int incy);
cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
cublasStatus_t cublasScopy(cublasHandle_t handle,
                           int n,
                           const float* x,
                           int incx,
                           float* y,
                           int incy);
cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);
cublasStatus_t cublasDaxpy(cublasHandle_t handle,
                           int n,
                           const double* alpha,
                           const double* x,
                           int incx,
                           double* y,
                           int incy);
cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);
cublasStatus_t cublasDcopy(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* y,
                           int incy);
cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);
cublasStatus_t cublasSdot(cublasHandle_t handle,
                          int n,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* result);
cublasStatus_t cublasDdot(cublasHandle_t handle,
                          int n,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* result);
cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result);
cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result);
cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result);
cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result);

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* b,
                           int ldb,
                           const float* beta,
                           float* c,
                           int ldc);

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* b,
                           int ldb,
                           const double* beta,
                           double* c,
                           int ldc);

#ifdef __cplusplus
}
#endif
