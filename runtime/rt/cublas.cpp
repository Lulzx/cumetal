#include "cublas_v2.h"

#include <cmath>
#include <mutex>
#include <new>

struct cublasContext {
    cudaStream_t stream = nullptr;
    std::mutex mutex;
};

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);

namespace {

constexpr int kCublasCompatVersion = 12000;

bool is_valid_operation(cublasOperation_t op) {
    return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

cublasStatus_t synchronize_handle_stream(cublasHandle_t handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const cudaError_t status = cudaStreamSynchronize(handle->stream);
    if (status != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasHandle_t created = new (std::nothrow) cublasContext();
    if (created == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    *handle = created;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version) {
    if (handle == nullptr || version == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    *version = kCublasCompatVersion;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream_id) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->stream = stream_id;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id) {
    if (handle == nullptr || stream_id == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *stream_id = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           const float* x,
                           int incx,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        y[i * incy] = alpha_value * x[i * incx] + y[i * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha_value;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle,
                           int n,
                           const double* alpha,
                           const double* x,
                           int incx,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        y[i * incy] = alpha_value * x[i * incx] + y[i * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha_value;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(cublasHandle_t handle,
                          int n,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
    }
    *result = static_cast<float>(sum);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(cublasHandle_t handle,
                          int n,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i * incx] * y[i * incy];
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
}

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
                           int ldc) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    const float beta_value = *beta;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                const float a_value = (transa == CUBLAS_OP_N) ? a[row + p * lda] : a[p + row * lda];
                const float b_value = (transb == CUBLAS_OP_N) ? b[p + col * ldb] : b[col + p * ldb];
                sum += a_value * b_value;
            }
            c[row + col * ldc] = alpha_value * sum + beta_value * c[row + col * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

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
                           int ldc) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                const double a_value = (transa == CUBLAS_OP_N) ? a[row + p * lda] : a[p + row * lda];
                const double b_value = (transb == CUBLAS_OP_N) ? b[p + col * ldb] : b[col + p * ldb];
                sum += a_value * b_value;
            }
            c[row + col * ldc] = alpha_value * sum + beta_value * c[row + col * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
