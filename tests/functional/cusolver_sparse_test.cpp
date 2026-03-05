#include <cusolverSp.h>
#include <cusparse.h>
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

// SPD 3x3 matrix (lower triangular stored, symmetric):
// A = [4  2  0]
//     [2  5  1]
//     [0  1  6]
// CSR of full matrix:
// row 0: (0,4) (1,2)
// row 1: (0,2) (1,5) (2,1)
// row 2: (1,1) (2,6)

static void test_cholesky_solve() {
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int rowPtr[] = {0, 2, 5, 7};
    int colInd[] = {0, 1, 0, 1, 2, 1, 2};
    float values[] = {4.0f, 2.0f, 2.0f, 5.0f, 1.0f, 1.0f, 6.0f};
    float b[] = {8.0f, 11.0f, 8.0f};
    float x[3] = {};
    int singularity = 0;

    cusolverStatus_t st = cusolverSpScsrlsvchol(handle, 3, 7, descrA,
                                                 values, rowPtr, colInd, b,
                                                 0.0f, 0, x, &singularity);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "cholesky solve success");
    CHECK(singularity == -1, "no singularity");

    // A * x = b → x should be [1, 1, 1] (verify: 4+2=6? No.)
    // Let me recompute: A*[1,1,1] = [4+2, 2+5+1, 1+6] = [6, 8, 7] ≠ b
    // Pick b = A*[1,2,1] = [4+4, 2+10+1, 2+6] = [8, 13, 8]
    // Wait, let me just check that A*x ≈ b
    float Ax0 = 4*x[0] + 2*x[1];
    float Ax1 = 2*x[0] + 5*x[1] + 1*x[2];
    float Ax2 = 1*x[1] + 6*x[2];
    CHECK(std::fabs(Ax0 - b[0]) < 0.1f, "cholesky Ax[0]≈b[0]");
    CHECK(std::fabs(Ax1 - b[1]) < 0.1f, "cholesky Ax[1]≈b[1]");
    CHECK(std::fabs(Ax2 - b[2]) < 0.1f, "cholesky Ax[2]≈b[2]");

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}

static void test_qr_solve() {
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // Simple 2x2: A = [3 1; 1 2], b = [5, 5] → x = [1, 2]
    int rowPtr[] = {0, 2, 4};
    int colInd[] = {0, 1, 0, 1};
    float values[] = {3.0f, 1.0f, 1.0f, 2.0f};
    float b[] = {5.0f, 5.0f};
    float x[2] = {};
    int singularity = 0;

    cusolverStatus_t st = cusolverSpScsrlsvqr(handle, 2, 4, descrA,
                                               values, rowPtr, colInd, b,
                                               0.0f, 0, x, &singularity);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "qr solve success");
    CHECK(singularity == -1, "no singularity (qr)");
    CHECK(std::fabs(x[0] - 1.0f) < 1e-4f, "qr x[0]=1");
    CHECK(std::fabs(x[1] - 2.0f) < 1e-4f, "qr x[1]=2");

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}

static void test_handle_lifecycle() {
    cusolverSpHandle_t handle;
    cusolverStatus_t st = cusolverSpCreate(&handle);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "cusolverSp create");
    st = cusolverSpDestroy(handle);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "cusolverSp destroy");
}

int main() {
    test_handle_lifecycle();
    test_cholesky_solve();
    test_qr_solve();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
