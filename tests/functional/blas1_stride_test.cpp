#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

// Helper: allocate device memory and copy from host
static float* dev_float(const float* src, int count) {
    float* d;
    cudaMalloc(&d, count * sizeof(float));
    cudaMemcpy(d, src, count * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

static double* dev_double(const double* src, int count) {
    double* d;
    cudaMalloc(&d, count * sizeof(double));
    cudaMemcpy(d, src, count * sizeof(double), cudaMemcpyHostToDevice);
    return d;
}

static void test_saxpy_stride2() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float hx[] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float hy[] = {10.0f, 99.0f, 20.0f, 99.0f, 30.0f};
    float* dx = dev_float(hx, 5);
    float* dy = dev_float(hy, 5);
    float alpha = 2.0f;

    cublasSaxpy(handle, 3, &alpha, dx, 2, dy, 2);

    float out[5];
    cudaMemcpy(out, dy, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(std::fabs(out[0] - 12.0f) < 1e-5f, "saxpy stride=2 y[0]=12");
    CHECK(std::fabs(out[2] - 24.0f) < 1e-5f, "saxpy stride=2 y[2]=24");
    CHECK(std::fabs(out[4] - 36.0f) < 1e-5f, "saxpy stride=2 y[4]=36");
    CHECK(std::fabs(out[1] - 99.0f) < 1e-5f, "saxpy stride=2 gap y[1] untouched");
    CHECK(std::fabs(out[3] - 99.0f) < 1e-5f, "saxpy stride=2 gap y[3] untouched");

    cudaFree(dx); cudaFree(dy);
    cublasDestroy(handle);
}

static void test_sdot_stride() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float hx[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
    float hy[] = {4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 6.0f};
    float* dx = dev_float(hx, 5);
    float* dy = dev_float(hy, 7);
    float result = 0.0f;

    // x with incx=2: elements 1,2,3
    // y with incy=3: elements 4,5,6
    // dot = 1*4 + 2*5 + 3*6 = 32
    cublasSdot(handle, 3, dx, 2, dy, 3, &result);
    CHECK(std::fabs(result - 32.0f) < 1e-5f, "sdot incx=2 incy=3 = 32");

    cudaFree(dx); cudaFree(dy);
    cublasDestroy(handle);
}

static void test_sscal_stride() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float hx[] = {1.0f, 100.0f, 2.0f, 100.0f, 3.0f};
    float* dx = dev_float(hx, 5);
    float alpha = 3.0f;

    cublasSscal(handle, 3, &alpha, dx, 2);

    float out[5];
    cudaMemcpy(out, dx, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(std::fabs(out[0] - 3.0f) < 1e-5f, "sscal stride=2 x[0]=3");
    CHECK(std::fabs(out[2] - 6.0f) < 1e-5f, "sscal stride=2 x[2]=6");
    CHECK(std::fabs(out[4] - 9.0f) < 1e-5f, "sscal stride=2 x[4]=9");
    CHECK(std::fabs(out[1] - 100.0f) < 1e-5f, "sscal stride=2 gap untouched");

    cudaFree(dx);
    cublasDestroy(handle);
}

static void test_snrm2_stride() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float hx[] = {3.0f, 0.0f, 4.0f};
    float* dx = dev_float(hx, 3);
    float result = 0.0f;

    // incx=2: elements 3, 4 -> norm = sqrt(9+16) = 5
    cublasSnrm2(handle, 2, dx, 2, &result);
    CHECK(std::fabs(result - 5.0f) < 1e-5f, "snrm2 stride=2 = 5");

    cudaFree(dx);
    cublasDestroy(handle);
}

static void test_sasum_stride() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float hx[] = {-1.0f, 0.0f, 2.0f, 0.0f, -3.0f};
    float* dx = dev_float(hx, 5);
    float result = 0.0f;

    // incx=2: elements -1, 2, -3 -> asum = 1+2+3 = 6
    cublasSasum(handle, 3, dx, 2, &result);
    CHECK(std::fabs(result - 6.0f) < 1e-5f, "sasum stride=2 = 6");

    cudaFree(dx);
    cublasDestroy(handle);
}

static void test_daxpy_stride() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double hx[] = {1.0, 0.0, 2.0};
    double hy[] = {10.0, 0.0, 20.0};
    double* dx = dev_double(hx, 3);
    double* dy = dev_double(hy, 3);
    double alpha = 5.0;

    cublasDaxpy(handle, 2, &alpha, dx, 2, dy, 2);

    double out[3];
    cudaMemcpy(out, dy, 3 * sizeof(double), cudaMemcpyDeviceToHost);

    CHECK(std::fabs(out[0] - 15.0) < 1e-12, "daxpy stride=2 y[0]=15");
    CHECK(std::fabs(out[2] - 30.0) < 1e-12, "daxpy stride=2 y[2]=30");
    CHECK(std::fabs(out[1] - 0.0) < 1e-12, "daxpy stride=2 gap untouched");

    cudaFree(dx); cudaFree(dy);
    cublasDestroy(handle);
}

static void test_scopy_stride() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float hx[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
    float hy[7] = {};
    float* dx = dev_float(hx, 5);
    float* dy = dev_float(hy, 7);

    // Copy x[0],x[2],x[4] to y[0],y[3],y[6]
    cublasScopy(handle, 3, dx, 2, dy, 3);

    float out[7];
    cudaMemcpy(out, dy, 7 * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(std::fabs(out[0] - 1.0f) < 1e-5f, "scopy incx=2 incy=3 y[0]=1");
    CHECK(std::fabs(out[3] - 2.0f) < 1e-5f, "scopy incx=2 incy=3 y[3]=2");
    CHECK(std::fabs(out[6] - 3.0f) < 1e-5f, "scopy incx=2 incy=3 y[6]=3");

    cudaFree(dx); cudaFree(dy);
    cublasDestroy(handle);
}

int main() {
    test_saxpy_stride2();
    test_sdot_stride();
    test_sscal_stride();
    test_snrm2_stride();
    test_sasum_stride();
    test_daxpy_stride();
    test_scopy_stride();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
