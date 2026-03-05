#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

// Lower triangular 3x3:
// [2 0 0]
// [1 3 0]
// [4 2 5]
// Solve L*y = x where x = [4, 7, 26]
// Expected: y = [2, 5/3, (26-8-10/3)/5 = (26-8-3.333)/5 = 14.667/5 = 2.933...]
// Actually: y[0]=4/2=2, y[1]=(7-1*2)/3=5/3, y[2]=(26-4*2-2*(5/3))/5=(26-8-10/3)/5

static void test_spsv_lower_fp32() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int rowPtr[] = {0, 1, 3, 6};
    int colInd[] = {0, 0, 1, 0, 1, 2};
    float values[] = {2.0f, 1.0f, 3.0f, 4.0f, 2.0f, 5.0f};
    float x[] = {4.0f, 7.0f, 26.0f};
    float y[] = {0.0f, 0.0f, 0.0f};

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, 3, 3, 6,
                      rowPtr, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_32F);

    cusparseSpSVDescr_t spsvDescr;
    cusparseSpSV_createDescr(&spsvDescr);

    float alpha = 1.0f;
    size_t bufSize = 0;
    cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, vecY,
                            CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufSize);

    cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecX, vecY,
                           CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, nullptr);
    cusparseStatus_t st = cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, matA, vecX, vecY,
                                              CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);
    CHECK(st == CUSPARSE_STATUS_SUCCESS, "SpSV solve returns success");

    // y[0] = 4/2 = 2
    CHECK(std::fabs(y[0] - 2.0f) < 1e-5f, "SpSV y[0]=2");
    // y[1] = (7 - 1*2)/3 = 5/3
    CHECK(std::fabs(y[1] - 5.0f/3.0f) < 1e-5f, "SpSV y[1]=5/3");
    // y[2] = (26 - 4*2 - 2*(5/3))/5 = (26-8-10/3)/5 = (78/3-24/3-10/3)/5 = 44/(3*5) = 44/15
    CHECK(std::fabs(y[2] - 44.0f/15.0f) < 1e-4f, "SpSV y[2]=44/15");

    cusparseSpSV_destroyDescr(spsvDescr);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
}

static void test_spsv_alpha_scaling() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Identity matrix
    int rowPtr[] = {0, 1, 2, 3};
    int colInd[] = {0, 1, 2};
    float values[] = {1.0f, 1.0f, 1.0f};
    float x[] = {2.0f, 3.0f, 4.0f};
    float y[] = {0.0f, 0.0f, 0.0f};

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, 3, 3, 3, rowPtr, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_32F);

    cusparseSpSVDescr_t spsvDescr;
    cusparseSpSV_createDescr(&spsvDescr);

    float alpha = 3.0f;
    cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, vecY,
                        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);

    CHECK(std::fabs(y[0] - 6.0f) < 1e-5f, "SpSV alpha=3 y[0]=6");
    CHECK(std::fabs(y[1] - 9.0f) < 1e-5f, "SpSV alpha=3 y[1]=9");
    CHECK(std::fabs(y[2] - 12.0f) < 1e-5f, "SpSV alpha=3 y[2]=12");

    cusparseSpSV_destroyDescr(spsvDescr);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
}

int main() {
    test_spsv_lower_fp32();
    test_spsv_alpha_scaling();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
