#include "cublas_v2.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    if (cublasCreate(nullptr) != CUBLAS_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_NOT_INITIALIZED for null handle out ptr\n");
        return 1;
    }

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cublasCreate failed\n");
        return 1;
    }
    int cublas_version = 0;
    if (cublasGetVersion(handle, &cublas_version) != CUBLAS_STATUS_SUCCESS || cublas_version <= 0) {
        std::fprintf(stderr, "FAIL: cublasGetVersion failed\n");
        return 1;
    }
    if (cublasGetVersion(handle, nullptr) != CUBLAS_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_NOT_INITIALIZED for null version ptr\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSetStream failed\n");
        return 1;
    }
    cudaStream_t queried_stream = nullptr;
    if (cublasGetStream(handle, &queried_stream) != CUBLAS_STATUS_SUCCESS || queried_stream != stream) {
        std::fprintf(stderr, "FAIL: cublasGetStream mismatch\n");
        return 1;
    }

    constexpr int kVecCount = 2048;
    std::vector<float> host_x(kVecCount);
    std::vector<float> host_y(kVecCount);
    std::vector<float> expected_y(kVecCount);
    for (int i = 0; i < kVecCount; ++i) {
        host_x[i] = static_cast<float>((i * 7) % 29) * 0.25f;
        host_y[i] = static_cast<float>((i * 3) % 17) * 0.5f;
        expected_y[i] = host_y[i];
    }
    const float alpha_axpy = 1.75f;
    for (int i = 0; i < kVecCount; ++i) {
        expected_y[i] = alpha_axpy * host_x[i] + expected_y[i];
    }

    float* dev_x = nullptr;
    float* dev_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_x), host_x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_y), host_y.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SAXPY failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_x, host_x.data(), host_x.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_y, host_y.data(), host_y.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SAXPY failed\n");
        return 1;
    }

    if (cublasSaxpy(handle, kVecCount, &alpha_axpy, dev_x, 1, dev_y, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSaxpy failed\n");
        return 1;
    }
    if (cudaMemcpy(host_y.data(), dev_y, host_y.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SAXPY failed\n");
        return 1;
    }
    for (int i = 0; i < kVecCount; ++i) {
        if (!nearly_equal(host_y[i], expected_y[i])) {
            std::fprintf(stderr,
                         "FAIL: SAXPY mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_y[i]),
                         static_cast<double>(expected_y[i]));
            return 1;
        }
    }

    if (cublasSaxpy(handle, kVecCount, &alpha_axpy, host_x.data(), 1, dev_y, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host x pointer\n");
        return 1;
    }

    const float alpha_scal = -0.5f;
    std::vector<float> expected_scal = host_x;
    for (float& value : expected_scal) {
        value *= alpha_scal;
    }
    if (cublasSscal(handle, kVecCount, &alpha_scal, dev_x, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSscal failed\n");
        return 1;
    }
    if (cudaMemcpy(host_x.data(), dev_x, host_x.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SSCAL failed\n");
        return 1;
    }
    for (int i = 0; i < kVecCount; ++i) {
        if (!nearly_equal(host_x[i], expected_scal[i])) {
            std::fprintf(stderr,
                         "FAIL: SSCAL mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_x[i]),
                         static_cast<double>(expected_scal[i]));
            return 1;
        }
    }
    if (cublasSscal(handle, kVecCount, &alpha_scal, expected_scal.data(), 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host x in SSCAL\n");
        return 1;
    }
    if (cublasSscal(handle, kVecCount, nullptr, dev_x, 1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for null alpha in SSCAL\n");
        return 1;
    }

    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;
    std::vector<float> host_a(lda * k);
    std::vector<float> host_b(ldb * n);
    std::vector<float> host_c(ldc * n);
    std::vector<float> expected_c(ldc * n);

    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            host_a[row + col * lda] = 1.0f + static_cast<float>(row) * 0.5f +
                                      static_cast<float>(col) * 0.25f;
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            host_b[row + col * ldb] =
                0.5f + static_cast<float>(row + 1) * static_cast<float>(col + 2) * 0.125f;
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            host_c[row + col * ldc] = static_cast<float>((row + 1) * (col + 2)) * 0.1f;
            expected_c[row + col * ldc] = host_c[row + col * ldc];
        }
    }

    const float alpha_gemm = 1.25f;
    const float beta_gemm = 0.75f;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += host_a[row + p * lda] * host_b[p + col * ldb];
            }
            expected_c[row + col * ldc] = alpha_gemm * sum + beta_gemm * expected_c[row + col * ldc];
        }
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_a), host_a.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_b), host_b.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_c), host_c.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_c, host_c.data(), host_c.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SGEMM failed\n");
        return 1;
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha_gemm,
                    dev_a,
                    lda,
                    dev_b,
                    ldb,
                    &beta_gemm,
                    dev_c,
                    ldc) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemm failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_c.size(); ++i) {
        if (!nearly_equal(host_c[i], expected_c[i])) {
            std::fprintf(stderr,
                         "FAIL: SGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected_c[i]));
            return 1;
        }
    }

    constexpr int mt = 3;
    constexpr int nt = 2;
    constexpr int kt = 4;
    constexpr int lda_t = kt;  // A is stored as (kt x mt), then transposed in op(A)
    constexpr int ldb_t = kt;  // B is (kt x nt)
    constexpr int ldc_t = mt;  // C is (mt x nt)

    std::vector<float> host_at(lda_t * mt);
    std::vector<float> host_bt(ldb_t * nt);
    std::vector<float> host_ct(ldc_t * nt);
    std::vector<float> expected_ct(ldc_t * nt);

    for (int col = 0; col < mt; ++col) {
        for (int row = 0; row < kt; ++row) {
            host_at[row + col * lda_t] =
                0.2f + static_cast<float>(row + 1) * static_cast<float>(col + 1) * 0.1f;
        }
    }
    for (int col = 0; col < nt; ++col) {
        for (int row = 0; row < kt; ++row) {
            host_bt[row + col * ldb_t] =
                0.3f + static_cast<float>(row + 2) * static_cast<float>(col + 1) * 0.07f;
        }
    }
    for (int col = 0; col < nt; ++col) {
        for (int row = 0; row < mt; ++row) {
            host_ct[row + col * ldc_t] = static_cast<float>((row + col + 1) * 0.15f);
            expected_ct[row + col * ldc_t] = host_ct[row + col * ldc_t];
        }
    }

    const float alpha_t = 0.9f;
    const float beta_t = 0.4f;
    for (int col = 0; col < nt; ++col) {
        for (int row = 0; row < mt; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < kt; ++p) {
                const float a_value = host_at[p + row * lda_t];  // transposed access
                const float b_value = host_bt[p + col * ldb_t];
                sum += a_value * b_value;
            }
            expected_ct[row + col * ldc_t] = alpha_t * sum + beta_t * expected_ct[row + col * ldc_t];
        }
    }

    float* dev_at = nullptr;
    float* dev_bt = nullptr;
    float* dev_ct = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_at), host_at.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_bt), host_bt.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_ct), host_ct.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for transposed SGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_at, host_at.data(), host_at.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_bt, host_bt.data(), host_bt.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_ct, host_ct.data(), host_ct.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for transposed SGEMM failed\n");
        return 1;
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    mt,
                    nt,
                    kt,
                    &alpha_t,
                    dev_at,
                    lda_t,
                    dev_bt,
                    ldb_t,
                    &beta_t,
                    dev_ct,
                    ldc_t) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemm transposed case failed\n");
        return 1;
    }
    if (cudaMemcpy(host_ct.data(), dev_ct, host_ct.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for transposed SGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_ct.size(); ++i) {
        if (!nearly_equal(host_ct[i], expected_ct[i])) {
            std::fprintf(stderr,
                         "FAIL: transposed SGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_ct[i]),
                         static_cast<double>(expected_ct[i]));
            return 1;
        }
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    mt,
                    nt,
                    kt,
                    &alpha_t,
                    dev_at,
                    mt - 1,
                    dev_bt,
                    ldb_t,
                    &beta_t,
                    dev_ct,
                    ldc_t) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid lda\n");
        return 1;
    }

    constexpr int mn = 2;
    constexpr int nn = 3;
    constexpr int kn = 4;
    constexpr int lda_n = mn;  // A is (mn x kn)
    constexpr int ldb_n = nn;  // B is stored as (nn x kn), then transposed in op(B)
    constexpr int ldc_n = mn;  // C is (mn x nn)

    std::vector<float> host_an(lda_n * kn);
    std::vector<float> host_bn(ldb_n * kn);
    std::vector<float> host_cn(ldc_n * nn);
    std::vector<float> expected_cn(ldc_n * nn);

    for (int col = 0; col < kn; ++col) {
        for (int row = 0; row < mn; ++row) {
            host_an[row + col * lda_n] =
                0.4f + static_cast<float>(row + 1) * static_cast<float>(col + 1) * 0.09f;
        }
    }
    for (int col = 0; col < kn; ++col) {
        for (int row = 0; row < nn; ++row) {
            host_bn[row + col * ldb_n] =
                0.1f + static_cast<float>(row + 2) * static_cast<float>(col + 1) * 0.06f;
        }
    }
    for (int col = 0; col < nn; ++col) {
        for (int row = 0; row < mn; ++row) {
            host_cn[row + col * ldc_n] = static_cast<float>((row + 1) * (col + 1)) * 0.2f;
            expected_cn[row + col * ldc_n] = host_cn[row + col * ldc_n];
        }
    }

    const float alpha_nt = 1.1f;
    const float beta_nt = 0.3f;
    for (int col = 0; col < nn; ++col) {
        for (int row = 0; row < mn; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < kn; ++p) {
                const float a_value = host_an[row + p * lda_n];
                const float b_value = host_bn[col + p * ldb_n];  // transposed access
                sum += a_value * b_value;
            }
            expected_cn[row + col * ldc_n] = alpha_nt * sum + beta_nt * expected_cn[row + col * ldc_n];
        }
    }

    float* dev_an = nullptr;
    float* dev_bn = nullptr;
    float* dev_cn = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_an), host_an.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_bn), host_bn.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_cn), host_cn.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for N,T SGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_an, host_an.data(), host_an.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_bn, host_bn.data(), host_bn.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_cn, host_cn.data(), host_cn.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for N,T SGEMM failed\n");
        return 1;
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    mn,
                    nn,
                    kn,
                    &alpha_nt,
                    dev_an,
                    lda_n,
                    dev_bn,
                    ldb_n,
                    &beta_nt,
                    dev_cn,
                    ldc_n) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemm N,T case failed\n");
        return 1;
    }
    if (cudaMemcpy(host_cn.data(), dev_cn, host_cn.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for N,T SGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_cn.size(); ++i) {
        if (!nearly_equal(host_cn[i], expected_cn[i])) {
            std::fprintf(stderr,
                         "FAIL: N,T SGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_cn[i]),
                         static_cast<double>(expected_cn[i]));
            return 1;
        }
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    mn,
                    nn,
                    kn,
                    &alpha_nt,
                    dev_an,
                    lda_n,
                    dev_bn,
                    nn - 1,
                    &beta_nt,
                    dev_cn,
                    ldc_n) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid ldb\n");
        return 1;
    }

    constexpr int md = 2;
    constexpr int nd = 2;
    constexpr int kd = 3;
    constexpr int lda_d = md;
    constexpr int ldb_d = kd;
    constexpr int ldc_d = md;
    std::vector<double> host_ad(lda_d * kd);
    std::vector<double> host_bd(ldb_d * nd);
    std::vector<double> host_cd(ldc_d * nd);
    std::vector<double> expected_cd(ldc_d * nd);

    for (int col = 0; col < kd; ++col) {
        for (int row = 0; row < md; ++row) {
            host_ad[row + col * lda_d] =
                0.25 + static_cast<double>(row + 1) * static_cast<double>(col + 1) * 0.05;
        }
    }
    for (int col = 0; col < nd; ++col) {
        for (int row = 0; row < kd; ++row) {
            host_bd[row + col * ldb_d] =
                0.5 + static_cast<double>(row + 1) * static_cast<double>(col + 2) * 0.03;
        }
    }
    for (int col = 0; col < nd; ++col) {
        for (int row = 0; row < md; ++row) {
            host_cd[row + col * ldc_d] = static_cast<double>((row + 1) * (col + 1)) * 0.2;
            expected_cd[row + col * ldc_d] = host_cd[row + col * ldc_d];
        }
    }

    const double alpha_d = 1.2;
    const double beta_d = 0.4;
    for (int col = 0; col < nd; ++col) {
        for (int row = 0; row < md; ++row) {
            double sum = 0.0;
            for (int p = 0; p < kd; ++p) {
                sum += host_ad[row + p * lda_d] * host_bd[p + col * ldb_d];
            }
            expected_cd[row + col * ldc_d] = alpha_d * sum + beta_d * expected_cd[row + col * ldc_d];
        }
    }

    double* dev_ad = nullptr;
    double* dev_bd = nullptr;
    double* dev_cd = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_ad), host_ad.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_bd), host_bd.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_cd), host_cd.size() * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_ad, host_ad.data(), host_ad.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_bd, host_bd.data(), host_bd.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_cd, host_cd.data(), host_cd.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DGEMM failed\n");
        return 1;
    }

    if (cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    md,
                    nd,
                    kd,
                    &alpha_d,
                    dev_ad,
                    lda_d,
                    dev_bd,
                    ldb_d,
                    &beta_d,
                    dev_cd,
                    ldc_d) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDgemm failed\n");
        return 1;
    }
    if (cudaMemcpy(host_cd.data(), dev_cd, host_cd.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_cd.size(); ++i) {
        if (std::fabs(host_cd[i] - expected_cd[i]) > 1e-9) {
            std::fprintf(stderr,
                         "FAIL: DGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         host_cd[i],
                         expected_cd[i]);
            return 1;
        }
    }
    if (cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    md,
                    nd,
                    kd,
                    &alpha_d,
                    host_ad.data(),
                    lda_d,
                    dev_bd,
                    ldb_d,
                    &beta_d,
                    dev_cd,
                    ldc_d) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host A in DGEMM\n");
        return 1;
    }

    constexpr int kDoubleCount = 1536;
    std::vector<double> host_dx(kDoubleCount);
    std::vector<double> host_dy(kDoubleCount);
    for (int i = 0; i < kDoubleCount; ++i) {
        host_dx[i] = 0.125 + static_cast<double>((i * 5) % 23) * 0.25;
        host_dy[i] = 0.5 + static_cast<double>((i * 3) % 17) * 0.1;
    }
    std::vector<double> expected_daxpy = host_dy;
    const double alpha_daxpy = -1.3;
    for (int i = 0; i < kDoubleCount; ++i) {
        expected_daxpy[i] = alpha_daxpy * host_dx[i] + expected_daxpy[i];
    }

    double* dev_dx = nullptr;
    double* dev_dy = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dx), host_dx.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dy), host_dy.size() * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DAXPY failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dx, host_dx.data(), host_dx.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_dy, host_dy.data(), host_dy.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DAXPY failed\n");
        return 1;
    }
    if (cublasDaxpy(handle, kDoubleCount, &alpha_daxpy, dev_dx, 1, dev_dy, 1) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDaxpy failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dy.data(), dev_dy, host_dy.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DAXPY failed\n");
        return 1;
    }
    for (int i = 0; i < kDoubleCount; ++i) {
        if (std::fabs(host_dy[i] - expected_daxpy[i]) > 1e-10) {
            std::fprintf(stderr,
                         "FAIL: DAXPY mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_dy[i],
                         expected_daxpy[i]);
            return 1;
        }
    }
    if (cublasDaxpy(handle, kDoubleCount, &alpha_daxpy, host_dx.data(), 1, dev_dy, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DAXPY\n");
        return 1;
    }

    const double alpha_dscal = 0.25;
    std::vector<double> expected_dscal = host_dx;
    for (double& value : expected_dscal) {
        value *= alpha_dscal;
    }
    if (cublasDscal(handle, kDoubleCount, &alpha_dscal, dev_dx, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDscal failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dx.data(), dev_dx, host_dx.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DSCAL failed\n");
        return 1;
    }
    for (int i = 0; i < kDoubleCount; ++i) {
        if (std::fabs(host_dx[i] - expected_dscal[i]) > 1e-10) {
            std::fprintf(stderr,
                         "FAIL: DSCAL mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_dx[i],
                         expected_dscal[i]);
            return 1;
        }
    }
    if (cublasDscal(handle, kDoubleCount, &alpha_dscal, expected_dscal.data(), 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DSCAL\n");
        return 1;
    }

    constexpr int kDotCount = 1024;
    std::vector<float> host_dot_x(kDotCount);
    std::vector<float> host_dot_y(kDotCount);
    for (int i = 0; i < kDotCount; ++i) {
        host_dot_x[i] = static_cast<float>((i * 2) % 31) * 0.2f;
        host_dot_y[i] = static_cast<float>((i * 5) % 19) * 0.3f;
    }
    float* dev_dot_x = nullptr;
    float* dev_dot_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dot_x), host_dot_x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dot_y), host_dot_y.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SDOT failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dot_x, host_dot_x.data(), host_dot_x.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_dot_y, host_dot_y.data(), host_dot_y.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SDOT failed\n");
        return 1;
    }

    float expected_sdot = 0.0f;
    for (int i = 0; i < kDotCount; ++i) {
        expected_sdot += host_dot_x[i] * host_dot_y[i];
    }
    float dot_result = 0.0f;
    if (cublasSdot(handle, kDotCount, dev_dot_x, 1, dev_dot_y, 1, &dot_result) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSdot failed\n");
        return 1;
    }
    if (std::fabs(dot_result - expected_sdot) > 5e-3f) {
        std::fprintf(stderr,
                     "FAIL: SDOT mismatch (got=%f expected=%f)\n",
                     static_cast<double>(dot_result),
                     static_cast<double>(expected_sdot));
        return 1;
    }
    if (cublasSdot(handle, kDotCount, host_dot_x.data(), 1, dev_dot_y, 1, &dot_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SDOT\n");
        return 1;
    }

    std::vector<double> host_ddot_x(kDotCount);
    std::vector<double> host_ddot_y(kDotCount);
    for (int i = 0; i < kDotCount; ++i) {
        host_ddot_x[i] = 0.1 + static_cast<double>((i * 7) % 29) * 0.2;
        host_ddot_y[i] = 0.2 + static_cast<double>((i * 11) % 17) * 0.15;
    }
    double* dev_ddot_x = nullptr;
    double* dev_ddot_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_ddot_x), host_ddot_x.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_ddot_y), host_ddot_y.size() * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DDOT failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_ddot_x,
                   host_ddot_x.data(),
                   host_ddot_x.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_ddot_y,
                   host_ddot_y.data(),
                   host_ddot_y.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DDOT failed\n");
        return 1;
    }

    double expected_ddot = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_ddot += host_ddot_x[i] * host_ddot_y[i];
    }
    double ddot_result = 0.0;
    if (cublasDdot(handle, kDotCount, dev_ddot_x, 1, dev_ddot_y, 1, &ddot_result) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDdot failed\n");
        return 1;
    }
    if (std::fabs(ddot_result - expected_ddot) > 1e-9) {
        std::fprintf(stderr, "FAIL: DDOT mismatch (got=%f expected=%f)\n", ddot_result, expected_ddot);
        return 1;
    }
    if (cublasDdot(handle, kDotCount, dev_ddot_x, 1, host_ddot_y.data(), 1, &ddot_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host Y in DDOT\n");
        return 1;
    }

    double expected_snrm2_acc = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_snrm2_acc += static_cast<double>(host_dot_x[i]) * static_cast<double>(host_dot_x[i]);
    }
    const float expected_snrm2 = static_cast<float>(std::sqrt(expected_snrm2_acc));
    float snrm2_result = 0.0f;
    if (cublasSnrm2(handle, kDotCount, dev_dot_x, 1, &snrm2_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSnrm2 failed\n");
        return 1;
    }
    if (std::fabs(snrm2_result - expected_snrm2) > 1e-4f) {
        std::fprintf(stderr,
                     "FAIL: SNRM2 mismatch (got=%f expected=%f)\n",
                     static_cast<double>(snrm2_result),
                     static_cast<double>(expected_snrm2));
        return 1;
    }
    if (cublasSnrm2(handle, kDotCount, host_dot_x.data(), 1, &snrm2_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SNRM2\n");
        return 1;
    }

    double expected_dnrm2_acc = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_dnrm2_acc += host_ddot_x[i] * host_ddot_x[i];
    }
    const double expected_dnrm2 = std::sqrt(expected_dnrm2_acc);
    double dnrm2_result = 0.0;
    if (cublasDnrm2(handle, kDotCount, dev_ddot_x, 1, &dnrm2_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDnrm2 failed\n");
        return 1;
    }
    if (std::fabs(dnrm2_result - expected_dnrm2) > 1e-10) {
        std::fprintf(stderr, "FAIL: DNRM2 mismatch (got=%f expected=%f)\n", dnrm2_result, expected_dnrm2);
        return 1;
    }
    if (cublasDnrm2(handle, kDotCount, host_ddot_x.data(), 1, &dnrm2_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DNRM2\n");
        return 1;
    }

    if (cudaFree(dev_x) != cudaSuccess || cudaFree(dev_y) != cudaSuccess ||
        cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess || cudaFree(dev_at) != cudaSuccess ||
        cudaFree(dev_bt) != cudaSuccess || cudaFree(dev_ct) != cudaSuccess ||
        cudaFree(dev_an) != cudaSuccess || cudaFree(dev_bn) != cudaSuccess ||
        cudaFree(dev_cn) != cudaSuccess || cudaFree(dev_ad) != cudaSuccess ||
        cudaFree(dev_bd) != cudaSuccess || cudaFree(dev_cd) != cudaSuccess ||
        cudaFree(dev_dx) != cudaSuccess || cudaFree(dev_dy) != cudaSuccess ||
        cudaFree(dev_dot_x) != cudaSuccess || cudaFree(dev_dot_y) != cudaSuccess ||
        cudaFree(dev_ddot_x) != cudaSuccess || cudaFree(dev_ddot_y) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDestroy failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuBLAS shim SAXPY/SGEMM operations validated\n");
    return 0;
}
