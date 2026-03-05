#include <cudnn.h>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_attn_descriptor_lifecycle() {
    cudnnAttnDescriptor_t attnDesc;
    cudnnStatus_t st = cudnnCreateAttnDescriptor(&attnDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "create attn descriptor");

    st = cudnnSetAttnDescriptor(attnDesc,
                                 0,    // attnMode
                                 8,    // nHeads
                                 1.0,  // smScaler
                                 CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH,
                                 nullptr, nullptr, // dropout descs
                                 64, 64, 64,       // qSize, kSize, vSize
                                 0, 0, 0, 0,       // proj sizes (0 = no projection)
                                 128, 128,         // max seq lengths
                                 32, 1);           // max batch, beam
    CHECK(st == CUDNN_STATUS_SUCCESS, "set attn descriptor");

    st = cudnnDestroyAttnDescriptor(attnDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "destroy attn descriptor");
}

static void test_attn_buffer_sizes() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnAttnDescriptor_t attnDesc;
    cudnnCreateAttnDescriptor(&attnDesc);
    cudnnSetAttnDescriptor(attnDesc, 0, 4, 1.0,
                            CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                            CUDNN_DEFAULT_MATH,
                            nullptr, nullptr,
                            32, 32, 32,   // q/k/v size
                            0, 0, 0, 0,   // no projections
                            64, 64, 16, 1);

    size_t weightSize = 0, workSize = 0, reserveSize = 0;
    cudnnStatus_t st = cudnnGetMultiHeadAttnBuffers(handle, attnDesc,
                                                     &weightSize, &workSize, &reserveSize);
    CHECK(st == CUDNN_STATUS_SUCCESS, "get attn buffers");
    CHECK(weightSize > 0, "attn weight size > 0");

    cudnnDestroyAttnDescriptor(attnDesc);
    cudnnDestroy(handle);
}

static void test_seq_data_descriptor() {
    cudnnSeqDataDescriptor_t seqDesc;
    cudnnStatus_t st = cudnnCreateSeqDataDescriptor(&seqDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "create seq data descriptor");

    int dims[] = {64, 32, 1, 128}; // time, batch, beam, vect
    cudnnSeqDataAxis_t axes[] = {CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM,
                                  CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
    int seqLengths[] = {64};
    st = cudnnSetSeqDataDescriptor(seqDesc, CUDNN_DATA_FLOAT, 4, dims, axes, 1, seqLengths, nullptr);
    CHECK(st == CUDNN_STATUS_SUCCESS, "set seq data descriptor");

    st = cudnnDestroySeqDataDescriptor(seqDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "destroy seq data descriptor");
}

static void test_attn_weight_pointers() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnAttnDescriptor_t attnDesc;
    cudnnCreateAttnDescriptor(&attnDesc);
    cudnnSetAttnDescriptor(attnDesc, 0, 2, 1.0,
                            CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                            CUDNN_DEFAULT_MATH,
                            nullptr, nullptr,
                            16, 16, 16,
                            0, 0, 0, 0,
                            32, 32, 8, 1);

    size_t weightSize = 0;
    cudnnGetMultiHeadAttnBuffers(handle, attnDesc, &weightSize, nullptr, nullptr);

    // Allocate dummy weights
    float* weights = new float[weightSize / sizeof(float)]();
    void* qAddr = nullptr;
    void* kAddr = nullptr;

    cudnnStatus_t st = cudnnGetMultiHeadAttnWeights(handle, attnDesc,
                                                     CUDNN_MH_ATTN_Q_WEIGHTS,
                                                     weightSize, weights,
                                                     nullptr, &qAddr);
    CHECK(st == CUDNN_STATUS_SUCCESS, "get Q weights");
    CHECK(qAddr == weights, "Q weights at start");

    st = cudnnGetMultiHeadAttnWeights(handle, attnDesc,
                                       CUDNN_MH_ATTN_K_WEIGHTS,
                                       weightSize, weights,
                                       nullptr, &kAddr);
    CHECK(st == CUDNN_STATUS_SUCCESS, "get K weights");
    CHECK(kAddr > qAddr, "K weights after Q weights");

    delete[] weights;
    cudnnDestroyAttnDescriptor(attnDesc);
    cudnnDestroy(handle);
}

int main() {
    test_attn_descriptor_lifecycle();
    test_attn_buffer_sizes();
    test_seq_data_descriptor();
    test_attn_weight_pointers();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
