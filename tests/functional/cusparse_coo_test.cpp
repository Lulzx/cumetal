#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

// 3x3 COO matrix (same as CSR tests):
// [1 0 2]
// [0 3 0]
// [4 0 5]
// rowInd=[0,0,1,2,2], colInd=[0,2,1,0,2], values=[1,2,3,4,5]

static void test_coo_spmv() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int rowInd[] = {0, 0, 1, 2, 2};
    int colInd[] = {0, 2, 1, 0, 2};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {0.0f, 0.0f, 0.0f};

    cusparseSpMatDescr_t matA;
    cusparseCreateCoo(&matA, 3, 3, 5,
                      rowInd, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    cusparseStatus_t st = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY,
                                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);
    CHECK(st == CUSPARSE_STATUS_SUCCESS, "COO SpMV returns success");
    CHECK(std::fabs(y[0] - 7.0f) < 1e-5f, "COO SpMV y[0]=7");
    CHECK(std::fabs(y[1] - 6.0f) < 1e-5f, "COO SpMV y[1]=6");
    CHECK(std::fabs(y[2] - 19.0f) < 1e-5f, "COO SpMV y[2]=19");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
}

static void test_coo_spmm() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int rowInd[] = {0, 0, 1, 2, 2};
    int colInd[] = {0, 2, 1, 0, 2};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    // Dense 3x2 B (col-major)
    float B[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float C[] = {0, 0, 0, 0, 0, 0};

    cusparseSpMatDescr_t matA;
    cusparseCreateCoo(&matA, 3, 3, 5,
                      rowInd, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t matB, matC;
    cusparseCreateDnMat(&matB, 3, 2, 3, B, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, 3, 2, 3, C, CUDA_R_32F, CUSPARSE_ORDER_COL);

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC,
                 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, nullptr);

    CHECK(std::fabs(C[0] - 7.0f) < 1e-5f, "COO SpMM C[0,0]=7");
    CHECK(std::fabs(C[1] - 6.0f) < 1e-5f, "COO SpMM C[1,0]=6");
    CHECK(std::fabs(C[2] - 19.0f) < 1e-5f, "COO SpMM C[2,0]=19");
    CHECK(std::fabs(C[3] - 16.0f) < 1e-5f, "COO SpMM C[0,1]=16");
    CHECK(std::fabs(C[4] - 15.0f) < 1e-5f, "COO SpMM C[1,1]=15");
    CHECK(std::fabs(C[5] - 46.0f) < 1e-5f, "COO SpMM C[2,1]=46");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
}

int main() {
    test_coo_spmv();
    test_coo_spmm();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
