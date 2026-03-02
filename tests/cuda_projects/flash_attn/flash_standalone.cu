// flash_standalone.cu — tspeterkim/flash-attention-minimal run standalone
//
// Original flash.cu stripped of PyTorch dependency; replaced with a plain
// CUDA host harness that allocates float buffers, calls the kernel, and
// verifies the result against a CPU reference softmax-attention.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Original kernel (verbatim from flash.cu, torch include removed) ───────────

__global__
void forward_kernel(const float* Q, const float* K, const float* V,
                    const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br,
                    const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset  = (bx * gridDim.y * N)     + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            for (int x = 0; x < d; x++)
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            float row_m = -3.402823466e+38f;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                if (sum > row_m) row_m = sum;
            }

            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            float row_m_new = (row_m_prev > row_m) ? row_m_prev : row_m;
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev)
                            + (__expf(row_m      - row_m_new) * row_l);

            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++)
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1.0f / row_l_new)
                    * ((row_l_prev * __expf(row_m_prev - row_m_new)
                        * O[qkv_offset + (tile_size * i) + (tx * d) + x])
                       + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

// ── CPU reference: standard scaled dot-product attention ──────────────────────

static void attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int B, int nh, int N, int d) {
    float* scores = (float*)malloc(N * N * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            int offset = (b * nh * N * d) + (h * N * d);
            // S = Q K^T * scale
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float s = 0;
                    for (int k = 0; k < d; k++)
                        s += Q[offset + i*d + k] * K[offset + j*d + k];
                    scores[i*N + j] = s * scale;
                }
            }
            // softmax rows
            for (int i = 0; i < N; i++) {
                float mx = scores[i*N];
                for (int j = 1; j < N; j++) if (scores[i*N+j] > mx) mx = scores[i*N+j];
                float s = 0;
                for (int j = 0; j < N; j++) { scores[i*N+j] = expf(scores[i*N+j]-mx); s += scores[i*N+j]; }
                for (int j = 0; j < N; j++) scores[i*N+j] /= s;
            }
            // O = softmax(S) V
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < d; k++) {
                    float v = 0;
                    for (int j = 0; j < N; j++)
                        v += scores[i*N+j] * V[offset + j*d + k];
                    O[offset + i*d + k] = v;
                }
            }
        }
    }
    free(scores);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    // Small test: B=1, nh=2, N=32, d=16 (fits comfortably in shared memory)
    const int B = 1, nh = 2, N = 32, d = 16;
    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float scale = 1.0f / sqrtf((float)d);
    const float TOL = 2e-5f;

    size_t qkvo_sz = (size_t)B * nh * N * d * sizeof(float);
    size_t lm_sz   = (size_t)B * nh * N     * sizeof(float);

    float* h_Q = (float*)malloc(qkvo_sz);
    float* h_K = (float*)malloc(qkvo_sz);
    float* h_V = (float*)malloc(qkvo_sz);
    float* h_O = (float*)calloc(B*nh*N*d, sizeof(float));
    float* h_ref = (float*)malloc(qkvo_sz);
    float* h_l = (float*)calloc(B*nh*N, sizeof(float));
    float* h_m = (float*)malloc(lm_sz);

    srand(1234);
    for (int i = 0; i < (int)(qkvo_sz/sizeof(float)); i++) {
        h_Q[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        h_K[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        h_V[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    }
    // Init m to -inf
    for (int i = 0; i < B*nh*N; i++) h_m[i] = -3.402823466e+38f;

    // CPU reference
    attention_cpu(h_Q, h_K, h_V, h_ref, B, nh, N, d);

    // CUDA
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc(&d_Q, qkvo_sz);  cudaMalloc(&d_K, qkvo_sz);
    cudaMalloc(&d_V, qkvo_sz);  cudaMalloc(&d_O, qkvo_sz);
    cudaMalloc(&d_l, lm_sz);    cudaMalloc(&d_m, lm_sz);

    cudaMemcpy(d_Q, h_Q, qkvo_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkvo_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkvo_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, qkvo_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, lm_sz,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, lm_sz,   cudaMemcpyHostToDevice);

    int sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);
    dim3 grid(B, nh);
    forward_kernel<<<grid, Bc, sram_size>>>(
        d_Q, d_K, d_V, N, d, Tc, Tr, Bc, Br, scale, d_l, d_m, d_O);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, qkvo_sz, cudaMemcpyDeviceToHost);

    // Verify
    int errs = 0;
    for (int i = 0; i < B*nh*N*d; i++) {
        float diff = fabsf(h_O[i] - h_ref[i]);
        if (diff > TOL) {
            if (errs < 4)
                printf("  [%d]: got %.6f ref %.6f diff %.2e\n", i, h_O[i], h_ref[i], diff);
            errs++;
        }
    }
    if (errs) {
        printf("FAIL: flash attention: %d errors\n", errs);
        return 1;
    }
    printf("PASS: flash-attention-minimal (B=%d nh=%d N=%d d=%d)\n", B, nh, N, d);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref); free(h_l); free(h_m);
    return 0;
}
