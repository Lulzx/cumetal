#include "cusolverDn.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include "cuda_runtime.h"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

// ── cuSOLVER shim ───────────────────────────────────────────────────────────
// Dense linear algebra via Apple Accelerate LAPACK.
// On Apple Silicon UMA, device pointers are host-accessible, so LAPACK
// operates directly on the caller's buffers with zero copy.

extern "C" {

struct cusolverDnContext {
    cudaStream_t stream = nullptr;
};

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    *handle = new cusolverDnContext();
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    delete handle;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (streamId) *streamId = handle->stream;
    return CUSOLVER_STATUS_SUCCESS;
}

static void sync_stream(cusolverDnHandle_t handle) {
    if (handle && handle->stream) cudaStreamSynchronize(handle->stream);
}

// ── LU factorization ────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                              float* /*A*/, int /*lda*/, int* Lwork) {
    if (Lwork) *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                              double* /*A*/, int /*lda*/, int* Lwork) {
    if (Lwork) *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n,
                                   float* A, int lda, float* /*Workspace*/,
                                   int* devIpiv, int* devInfo) {
    sync_stream(handle);
    __CLPK_integer M = m, N = n, LDA = lda, info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(std::min(m, n)));
    sgetrf_(&M, &N, A, &LDA, ipiv.data(), &info);
    if (devIpiv) {
        for (int i = 0; i < std::min(m, n); ++i) devIpiv[i] = static_cast<int>(ipiv[static_cast<size_t>(i)]);
    }
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n,
                                   double* A, int lda, double* /*Workspace*/,
                                   int* devIpiv, int* devInfo) {
    sync_stream(handle);
    __CLPK_integer M = m, N = n, LDA = lda, info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(std::min(m, n)));
    dgetrf_(&M, &N, A, &LDA, ipiv.data(), &info);
    if (devIpiv) {
        for (int i = 0; i < std::min(m, n); ++i) devIpiv[i] = static_cast<int>(ipiv[static_cast<size_t>(i)]);
    }
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── LU solve ─────────────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const float* A, int lda,
                                   const int* devIpiv, float* B, int ldb,
                                   int* devInfo) {
    sync_stream(handle);
    char t = (trans == 0) ? 'N' : 'T';
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) ipiv[static_cast<size_t>(i)] = devIpiv[i];
    sgetrs_(&t, &N, &NRHS, const_cast<float*>(A), &LDA, ipiv.data(), B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const double* A, int lda,
                                   const int* devIpiv, double* B, int ldb,
                                   int* devInfo) {
    sync_stream(handle);
    char t = (trans == 0) ? 'N' : 'T';
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) ipiv[static_cast<size_t>(i)] = devIpiv[i];
    dgetrs_(&t, &N, &NRHS, const_cast<double*>(A), &LDA, ipiv.data(), B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── QR factorization ────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                              float* /*A*/, int /*lda*/, int* Lwork) {
    if (Lwork) *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                              double* /*A*/, int /*lda*/, int* Lwork) {
    if (Lwork) *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n,
                                   float* A, int lda, float* TAU,
                                   float* Workspace, int Lwork, int* devInfo) {
    sync_stream(handle);
    __CLPK_integer M = m, N = n, LDA = lda, LW = Lwork, info = 0;
    sgeqrf_(&M, &N, A, &LDA, TAU, Workspace, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n,
                                   double* A, int lda, double* TAU,
                                   double* Workspace, int Lwork, int* devInfo) {
    sync_stream(handle);
    __CLPK_integer M = m, N = n, LDA = lda, LW = Lwork, info = 0;
    dgeqrf_(&M, &N, A, &LDA, TAU, Workspace, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── Cholesky factorization ──────────────────────────────────────────────────

cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t /*handle*/,
                                              cublasFillMode_t /*uplo*/, int n,
                                              float* /*A*/, int /*lda*/, int* Lwork) {
    if (Lwork) *Lwork = n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t /*handle*/,
                                              cublasFillMode_t /*uplo*/, int n,
                                              double* /*A*/, int /*lda*/, int* Lwork) {
    if (Lwork) *Lwork = n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, float* A, int lda, float* /*Workspace*/,
                                   int /*Lwork*/, int* devInfo) {
    sync_stream(handle);
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, info = 0;
    spotrf_(&ul, &N, A, &LDA, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, double* A, int lda, double* /*Workspace*/,
                                   int /*Lwork*/, int* devInfo) {
    sync_stream(handle);
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, info = 0;
    dpotrf_(&ul, &N, A, &LDA, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── Cholesky solve ──────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, int nrhs, const float* A, int lda,
                                   float* B, int ldb, int* devInfo) {
    sync_stream(handle);
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    spotrs_(&ul, &N, &NRHS, const_cast<float*>(A), &LDA, B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, int nrhs, const double* A, int lda,
                                   double* B, int ldb, int* devInfo) {
    sync_stream(handle);
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    dpotrs_(&ul, &N, &NRHS, const_cast<double*>(A), &LDA, B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── Eigenvalue decomposition (syevd) ────────────────────────────────────────

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t /*handle*/,
                                              cusolverEigMode_t /*jobz*/,
                                              cublasFillMode_t /*uplo*/, int n,
                                              const float* /*A*/, int /*lda*/,
                                              const float* /*W*/, int* lwork) {
    // Query optimal workspace
    if (lwork) *lwork = std::max(1, 1 + 6 * n + 2 * n * n);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t /*handle*/,
                                              cusolverEigMode_t /*jobz*/,
                                              cublasFillMode_t /*uplo*/, int n,
                                              const double* /*A*/, int /*lda*/,
                                              const double* /*W*/, int* lwork) {
    if (lwork) *lwork = std::max(1, 1 + 6 * n + 2 * n * n);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, float* A, int lda,
                                   float* W, float* work, int lwork, int* devInfo) {
    sync_stream(handle);
    char job = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? 'V' : 'N';
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, LW = lwork, info = 0;
    // LAPACK's ssyevd also needs integer workspace
    __CLPK_integer liwork = std::max(__CLPK_integer(1), 3 + 5 * N);
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(liwork));
    ssyevd_(&job, &ul, &N, A, &LDA, W, work, &LW, iwork.data(), &liwork, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, double* A, int lda,
                                   double* W, double* work, int lwork, int* devInfo) {
    sync_stream(handle);
    char job = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? 'V' : 'N';
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, LW = lwork, info = 0;
    __CLPK_integer liwork = std::max(__CLPK_integer(1), 3 + 5 * N);
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(liwork));
    dsyevd_(&job, &ul, &N, A, &LDA, W, work, &LW, iwork.data(), &liwork, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── SVD ─────────────────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                              int* lwork) {
    if (lwork) *lwork = std::max(1, 3 * std::min(m, n) + std::max(m, n) + std::max(m, n));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                              int* lwork) {
    if (lwork) *lwork = std::max(1, 3 * std::min(m, n) + std::max(m, n) + std::max(m, n));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu,
                                   signed char jobvt, int m, int n, float* A, int lda,
                                   float* S, float* U, int ldu, float* VT, int ldvt,
                                   float* work, int lwork, float* /*rwork*/, int* devInfo) {
    sync_stream(handle);
    char ju = static_cast<char>(jobu);
    char jvt = static_cast<char>(jobvt);
    __CLPK_integer M = m, N = n, LDA = lda, LDU = ldu, LDVT = ldvt, LW = lwork, info = 0;
    sgesvd_(&ju, &jvt, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, work, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu,
                                   signed char jobvt, int m, int n, double* A, int lda,
                                   double* S, double* U, int ldu, double* VT, int ldvt,
                                   double* work, int lwork, double* /*rwork*/, int* devInfo) {
    sync_stream(handle);
    char ju = static_cast<char>(jobu);
    char jvt = static_cast<char>(jobvt);
    __CLPK_integer M = m, N = n, LDA = lda, LDU = ldu, LDVT = ldvt, LW = lwork, info = 0;
    dgesvd_(&ju, &jvt, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, work, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

} // extern "C" — temporarily close for C++ templates

// ── cusolverSp: Sparse solver (host path) ─────────────────────────────────────
// Uses dense conversion + LAPACK as a simple-but-correct fallback.
// On UMA this is zero-copy from the caller's perspective.

// Helper: convert CSR to dense column-major matrix (must be outside extern "C")
template <typename T>
static void csr_to_dense_sp(int m, const T* csrVal, const int* csrRowPtr, const int* csrColInd,
                              int base, T* dense) {
    std::memset(dense, 0, (size_t)m * m * sizeof(T));
    for (int i = 0; i < m; ++i) {
        for (int j = csrRowPtr[i] - base; j < csrRowPtr[i + 1] - base; ++j) {
            const int c = csrColInd[j] - base;
            dense[(size_t)c * m + i] = csrVal[j];  // column-major
        }
    }
}

extern "C" {

struct cusolverSpContext {
    cudaStream_t stream = nullptr;
};

cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t* handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    *handle = new cusolverSpContext();
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle) {
    delete handle;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle, cudaStream_t streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUSOLVER_STATUS_SUCCESS;
}

// Sparse solve via dense Cholesky (LAPACK spotrf/dpotrf + spotrs/dpotrs)
cusolverStatus_t cusolverSpScsrlsvchol(cusolverSpHandle_t handle,
                                        int m, int /*nnz*/,
                                        const cusparseMatDescr_t descrA,
                                        const float* csrVal, const int* csrRowPtr,
                                        const int* csrColInd, const float* b,
                                        float /*tol*/, int /*reorder*/,
                                        float* x, int* singularity) {
    if (!handle || !csrVal || !csrRowPtr || !csrColInd || !b || !x)
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (handle->stream) cudaStreamSynchronize(handle->stream);
    const int base = descrA ? static_cast<int>(cusparseGetMatIndexBase(descrA)) : 0;

    std::vector<float> A(m * m);
    csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
    std::memcpy(x, b, m * sizeof(float));

    char uplo = 'L';
    __CLPK_integer N = m, nrhs = 1, lda = m, ldb = m, info = 0;
    spotrf_(&uplo, &N, A.data(), &lda, &info);
    if (info != 0) {
        if (singularity) *singularity = static_cast<int>(info - 1);
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
    spotrs_(&uplo, &N, &nrhs, A.data(), &lda, x, &ldb, &info);
    if (singularity) *singularity = -1; // no singularity
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverSpDcsrlsvchol(cusolverSpHandle_t handle,
                                        int m, int /*nnz*/,
                                        const cusparseMatDescr_t descrA,
                                        const double* csrVal, const int* csrRowPtr,
                                        const int* csrColInd, const double* b,
                                        double /*tol*/, int /*reorder*/,
                                        double* x, int* singularity) {
    if (!handle || !csrVal || !csrRowPtr || !csrColInd || !b || !x)
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (handle->stream) cudaStreamSynchronize(handle->stream);
    const int base = descrA ? static_cast<int>(cusparseGetMatIndexBase(descrA)) : 0;

    std::vector<double> A(m * m);
    csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
    std::memcpy(x, b, m * sizeof(double));

    char uplo = 'L';
    __CLPK_integer N = m, nrhs = 1, lda = m, ldb = m, info = 0;
    dpotrf_(&uplo, &N, A.data(), &lda, &info);
    if (info != 0) {
        if (singularity) *singularity = static_cast<int>(info - 1);
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
    dpotrs_(&uplo, &N, &nrhs, A.data(), &lda, x, &ldb, &info);
    if (singularity) *singularity = -1;
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// Sparse QR solve via dense QR (LAPACK sgels/dgels)
cusolverStatus_t cusolverSpScsrlsvqr(cusolverSpHandle_t handle,
                                      int m, int /*nnz*/,
                                      const cusparseMatDescr_t descrA,
                                      const float* csrVal, const int* csrRowPtr,
                                      const int* csrColInd, const float* b,
                                      float /*tol*/, int /*reorder*/,
                                      float* x, int* singularity) {
    if (!handle || !csrVal || !csrRowPtr || !csrColInd || !b || !x)
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (handle->stream) cudaStreamSynchronize(handle->stream);
    const int base = descrA ? static_cast<int>(cusparseGetMatIndexBase(descrA)) : 0;

    std::vector<float> A(m * m);
    csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
    std::memcpy(x, b, m * sizeof(float));

    char trans = 'N';
    __CLPK_integer M = m, N = m, nrhs = 1, lda = m, ldb = m, lwork = -1, info = 0;
    float work_query = 0;
    sgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, &work_query, &lwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(lwork);
    sgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, work.data(), &lwork, &info);
    if (singularity) *singularity = (info != 0) ? static_cast<int>(info - 1) : -1;
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverSpDcsrlsvqr(cusolverSpHandle_t handle,
                                      int m, int /*nnz*/,
                                      const cusparseMatDescr_t descrA,
                                      const double* csrVal, const int* csrRowPtr,
                                      const int* csrColInd, const double* b,
                                      double /*tol*/, int /*reorder*/,
                                      double* x, int* singularity) {
    if (!handle || !csrVal || !csrRowPtr || !csrColInd || !b || !x)
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (handle->stream) cudaStreamSynchronize(handle->stream);
    const int base = descrA ? static_cast<int>(cusparseGetMatIndexBase(descrA)) : 0;

    std::vector<double> A(m * m);
    csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
    std::memcpy(x, b, m * sizeof(double));

    char trans = 'N';
    __CLPK_integer M = m, N = m, nrhs = 1, lda = m, ldb = m, lwork = -1, info = 0;
    double work_query = 0;
    dgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, &work_query, &lwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<double> work(lwork);
    dgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, work.data(), &lwork, &info);
    if (singularity) *singularity = (info != 0) ? static_cast<int>(info - 1) : -1;
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

}  // extern "C"
