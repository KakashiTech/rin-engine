/*
 * RIN-X ULTIMATE - FASE 2: CUTLASS-STYLE KERNELS
 * GEMM kernels con epílogos fusionados, 2:4 sparsity, AMX/AVX-512
 */

#ifndef RINX_CUTLASS_KERNELS_H
#define RINX_CUTLASS_KERNELS_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// CONFIGURACIÓN Y TIPOS
// ============================================================================

#define RINX_CACHE_LINE 64
#define RINX_SIMD_WIDTH 16  // AVX-512: 16 floats
#define RINX_PAGE_SIZE 4096

// Epílogos fusionados
typedef enum {
    RINX_EPILOGUE_NONE = 0,
    RINX_EPILOGUE_BIAS = 1,
    RINX_EPILOGUE_RELU = 2,
    RINX_EPILOGUE_GELU = 4,
    RINX_EPILOGUE_SIGMOID = 8,
    RINX_EPILOGUE_BIAS_RELU = RINX_EPILOGUE_BIAS | RINX_EPILOGUE_RELU,
    RINX_EPILOGUE_BIAS_GELU = RINX_EPILOGUE_BIAS | RINX_EPILOGUE_GELU,
} rinx_epilogue_t;

// Formatos de sparsity
typedef enum {
    RINX_SPARSE_NONE = 0,
    RINX_SPARSE_2_4,      // 2:4 structured
    RINX_SPARSE_1_4,      // 1:4 structured  
    RINX_SPARSE_BLOCK_4x4, // Block 4x4
} rinx_sparsity_format_t;

// Algoritmos GEMM
typedef enum {
    RINX_GEMM_ALGO_DEFAULT = 0,
    RINX_GEMM_ALGO_TILED,
    RINX_GEMM_ALGO_STREAMK,  // CUTLASS Stream-K
    RINX_GEMM_ALGO_SPLIT_K,
} rinx_gemm_algo_t;

// ============================================================================
// ESTRUCTURAS DE ARGUMENTOS (CUTLASS-STYLE)
// ============================================================================

typedef struct {
    int M, N, K;
    float alpha, beta;
    const float* A;
    int lda;
    const float* B;
    int ldb;
    float* C;
    int ldc;
    const float* bias;
    rinx_epilogue_t epilogue;
    rinx_sparsity_format_t sparsity_A;
    rinx_sparsity_format_t sparsity_B;
    rinx_gemm_algo_t algo;
    int num_threads;
} rinx_gemm_args_t;

typedef struct {
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int groups;
    rinx_epilogue_t epilogue;
} rinx_conv_args_t;

// ============================================================================
// UTILIDADES SIMD
// ============================================================================

// Cargar alinear a 64 bytes
#define RINX_LOAD_ALIGNED(ptr) _mm512_load_ps(ptr)
#define RINX_STORE_ALIGNED(ptr, val) _mm512_store_ps(ptr, val)
#define RINX_LOAD_UNALIGNED(ptr) _mm512_loadu_ps(ptr)
#define RINX_STORE_UNALIGNED(ptr, val) _mm512_storeu_ps(ptr, val)

// FMA: a * b + c
#define RINX_FMA(a, b, c) _mm512_fmadd_ps(a, b, c)

// Broadcast
#define RINX_BROADCAST(val) _mm512_set1_ps(val)

// Zero
#define RINX_ZERO() _mm512_setzero_ps()

// Mask operations
#define RINX_MASK_LOAD(ptr, mask) _mm512_maskz_loadu_ps(mask, ptr)
#define RINX_MASK_STORE(ptr, mask, val) _mm512_mask_storeu_ps(ptr, mask, val)

// Comparaciones
#define RINX_CMP_GT(a, b) _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
#define RINX_CMP_LT(a, b) _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)

// Max/Min
#define RINX_MAX(a, b) _mm512_max_ps(a, b)
#define RINX_MIN(a, b) _mm512_min_ps(a, b)

// Epílogos vectorizados
static inline __m512 rinx_relu_ps(__m512 x) {
    return RINX_MAX(x, RINX_ZERO());
}

static inline __m512 rinx_gelu_ps(__m512 x) {
    // Aproximación: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    __m512 sqrt_2_over_pi = RINX_BROADCAST(0.7978845608f);
    __m512 coeff = RINX_BROADCAST(0.044715f);
    __m512 half = RINX_BROADCAST(0.5f);
    __m512 one = RINX_BROADCAST(1.0f);
    
    __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
    __m512 inner = RINX_FMA(x3, coeff, x);
    inner = _mm512_mul_ps(inner, sqrt_2_over_pi);
    
    // tanh aproximado (racional)
    __m512 x2 = _mm512_mul_ps(inner, inner);
    __m512 denom = RINX_FMA(x2, RINX_BROADCAST(0.142857f), one);
    __m512 tanh_approx = _mm512_div_ps(inner, denom);
    
    __m512 sigmoid_approx = RINX_FMA(tanh_approx, half, half);
    return _mm512_mul_ps(x, sigmoid_approx);
}

static inline __m512 rinx_sigmoid_ps(__m512 x) {
    // Aproximación: x / (1 + |x|) para x >= 0, más refinada
    __m512 one = RINX_BROADCAST(1.0f);
    __m512 abs_x = _mm512_abs_ps(x);
    __m512 denom = RINX_FMA(abs_x, RINX_BROADCAST(0.5f), one);
    __m512 sign = _mm512_div_ps(x, abs_x);  // 1 o -1
    return RINX_FMA(sign, RINX_BROADCAST(0.5f), RINX_BROADCAST(0.5f));
}

// Reducción horizontal
static inline float rinx_reduce_sum_ps(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

// ============================================================================
// TILING Y BLOCKING (CACHE-AWARE)
// ============================================================================

typedef struct {
    int Mt, Nt, Kt;      // Tile sizes
    int Mw, Nw, Kw;      // Warp sizes (micro-kernel)
    int Mz, Nz;          // Z-pipeline para double-buffering
} rinx_tile_config_t;

// Configuraciones óptimas por arquitectura
static const rinx_tile_config_t RINX_TILE_CONFIG_SKYLAKE = {
    .Mt = 128, .Nt = 128, .Kt = 256,
    .Mw = 8, .Nw = 16, .Kw = 1,
    .Mz = 2, .Nz = 2
};

static const rinx_tile_config_t RINX_TILE_CONFIG_SPR = {  // Sapphire Rapids con AMX
    .Mt = 256, .Nt = 256, .Kt = 512,
    .Mw = 16, .Nw = 16, .Kw = 1,
    .Mz = 4, .Nz = 4
};

static const rinx_tile_config_t RINX_TILE_CONFIG_AMD = {
    .Mt = 96, .Nt = 96, .Kt = 256,
    .Mw = 6, .Nw = 16, .Kw = 1,
    .Mz = 2, .Nz = 2
};

// ============================================================================
// MICRO-KERNELS GEMM (CORE COMPUTATION)
// ============================================================================

// Micro-kernel 8x16 (8 filas x 16 columnas AVX-512)
static inline void rinx_gemm_microkernel_8x16(const float* __restrict A,
                                               const float* __restrict B,
                                               float* __restrict C,
                                               int K, int ldc) {
    __m512 c0 = RINX_ZERO(), c1 = RINX_ZERO(), c2 = RINX_ZERO(), c3 = RINX_ZERO();
    __m512 c4 = RINX_ZERO(), c5 = RINX_ZERO(), c6 = RINX_ZERO(), c7 = RINX_ZERO();
    
    // Prefetch próximos datos
    _mm_prefetch((const char*)(A + 8 * K), _MM_HINT_T0);
    _mm_prefetch((const char*)(B + 16), _MM_HINT_T0);
    
    for (int k = 0; k < K; k++) {
        __m512 b = RINX_LOAD_UNALIGNED(B + k * 16);
        
        c0 = RINX_FMA(RINX_BROADCAST(A[0 * K + k]), b, c0);
        c1 = RINX_FMA(RINX_BROADCAST(A[1 * K + k]), b, c1);
        c2 = RINX_FMA(RINX_BROADCAST(A[2 * K + k]), b, c2);
        c3 = RINX_FMA(RINX_BROADCAST(A[3 * K + k]), b, c3);
        c4 = RINX_FMA(RINX_BROADCAST(A[4 * K + k]), b, c4);
        c5 = RINX_FMA(RINX_BROADCAST(A[5 * K + k]), b, c5);
        c6 = RINX_FMA(RINX_BROADCAST(A[6 * K + k]), b, c6);
        c7 = RINX_FMA(RINX_BROADCAST(A[7 * K + k]), b, c7);
    }
    
    // Store con epílogo opcional (template-like)
    RINX_STORE_UNALIGNED(C + 0 * ldc, c0);
    RINX_STORE_UNALIGNED(C + 1 * ldc, c1);
    RINX_STORE_UNALIGNED(C + 2 * ldc, c2);
    RINX_STORE_UNALIGNED(C + 3 * ldc, c3);
    RINX_STORE_UNALIGNED(C + 4 * ldc, c4);
    RINX_STORE_UNALIGNED(C + 5 * ldc, c5);
    RINX_STORE_UNALIGNED(C + 6 * ldc, c6);
    RINX_STORE_UNALIGNED(C + 7 * ldc, c7);
}

// Micro-kernel 8x16 con epílogo BIAS+RELU fusionado
static inline void rinx_gemm_microkernel_8x16_bias_relu(const float* __restrict A,
                                                         const float* __restrict B,
                                                         const float* __restrict bias,
                                                         float* __restrict C,
                                                         int K, int ldc, int n) {
    __m512 c0 = RINX_ZERO(), c1 = RINX_ZERO(), c2 = RINX_ZERO(), c3 = RINX_ZERO();
    __m512 c4 = RINX_ZERO(), c5 = RINX_ZERO(), c6 = RINX_ZERO(), c7 = RINX_ZERO();
    
    // Computación GEMM
    for (int k = 0; k < K; k++) {
        __m512 b = RINX_LOAD_UNALIGNED(B + k * 16);
        
        c0 = RINX_FMA(RINX_BROADCAST(A[0 * K + k]), b, c0);
        c1 = RINX_FMA(RINX_BROADCAST(A[1 * K + k]), b, c1);
        c2 = RINX_FMA(RINX_BROADCAST(A[2 * K + k]), b, c2);
        c3 = RINX_FMA(RINX_BROADCAST(A[3 * K + k]), b, c3);
        c4 = RINX_FMA(RINX_BROADCAST(A[4 * K + k]), b, c4);
        c5 = RINX_FMA(RINX_BROADCAST(A[5 * K + k]), b, c5);
        c6 = RINX_FMA(RINX_BROADCAST(A[6 * K + k]), b, c6);
        c7 = RINX_FMA(RINX_BROADCAST(A[7 * K + k]), b, c7);
    }
    
    // Epílogo fusionado: Bias + ReLU
    __m512 b0 = RINX_LOAD_UNALIGNED(bias + 0);
    __m512 b1 = RINX_LOAD_UNALIGNED(bias + 0); // Mismo bias para todas las filas
    
    c0 = rinx_relu_ps(RINX_FMA(c0, RINX_BROADCAST(1.0f), b0));
    c1 = rinx_relu_ps(RINX_FMA(c1, RINX_BROADCAST(1.0f), b1));
    c2 = rinx_relu_ps(RINX_FMA(c2, RINX_BROADCAST(1.0f), b0));
    c3 = rinx_relu_ps(RINX_FMA(c3, RINX_BROADCAST(1.0f), b1));
    c4 = rinx_relu_ps(RINX_FMA(c4, RINX_BROADCAST(1.0f), b0));
    c5 = rinx_relu_ps(RINX_FMA(c5, RINX_BROADCAST(1.0f), b1));
    c6 = rinx_relu_ps(RINX_FMA(c6, RINX_BROADCAST(1.0f), b0));
    c7 = rinx_relu_ps(RINX_FMA(c7, RINX_BROADCAST(1.0f), b1));
    
    // Store con mask para últimas columnas
    __mmask16 mask = (n >= 16) ? 0xFFFF : ((1 << n) - 1);
    RINX_MASK_STORE(C + 0 * ldc, mask, c0);
    RINX_MASK_STORE(C + 1 * ldc, mask, c1);
    RINX_MASK_STORE(C + 2 * ldc, mask, c2);
    RINX_MASK_STORE(C + 3 * ldc, mask, c3);
    RINX_MASK_STORE(C + 4 * ldc, mask, c4);
    RINX_MASK_STORE(C + 5 * ldc, mask, c5);
    RINX_MASK_STORE(C + 6 * ldc, mask, c6);
    RINX_MASK_STORE(C + 7 * ldc, mask, c7);
}

// ============================================================================
// 2:4 STRUCTURED SPARSITY KERNELS
// ============================================================================

// Estructura para matriz 2:4 sparse
typedef struct {
    const float* values;     // Non-zero values (50% del original)
    const uint8_t* metadata; // 2 bits por bloque de 4 (0-3 para posiciones)
    int rows, cols;
    int nnz;  // Number of non-zeros = rows * cols / 2
} rinx_sparse_2_4_matrix_t;

// Descompresión 2:4 on-the-fly durante GEMM
// Metadata: 2 bits indican cuáles 2 de 4 son non-zero
// 00: elementos 0,1 | 01: 0,2 | 10: 0,3 | 11: 1,2 | etc.

static inline void rinx_gemm_sparse_2_4_microkernel(const float* __restrict A,  // Dense MxK
                                                     const rinx_sparse_2_4_matrix_t* B, // Sparse 2:4 KxN
                                                     float* __restrict C,
                                                     int M, int N, int K,
                                                     int m_start, int n_start) {
    // 2 filas x 16 columnas
    __m512 c0 = RINX_ZERO(), c1 = RINX_ZERO();
    
    const float* A_row0 = A + (m_start + 0) * K;
    const float* A_row1 = A + (m_start + 1) * K;
    
    for (int k = 0; k < K; k += 4) {
        // Leer metadata para este bloque de 4
        int meta_idx = (k / 4) * (N / 4) + (n_start / 4);
        uint8_t meta = B->metadata[meta_idx];
        
        // Decodificar posiciones de los 2 non-zeros
        // Simplificado: asumimos orden estándar
        int pos0 = (meta >> 2) & 0x3;
        int pos1 = meta & 0x3;
        
        // Cargar los 2 valores non-zero
        float b_val0 = B->values[(k / 2) * N + n_start + pos0];  // Simplificado
        float b_val1 = B->values[(k / 2) * N + n_start + pos1];
        
        // Broadcast de B
        __m512 b0 = RINX_BROADCAST(b_val0);
        __m512 b1 = RINX_BROADCAST(b_val1);
        
        // FMA con A
        c0 = RINX_FMA(RINX_BROADCAST(A_row0[k + pos0]), b0, c0);
        c0 = RINX_FMA(RINX_BROADCAST(A_row0[k + pos1]), b1, c0);
        c1 = RINX_FMA(RINX_BROADCAST(A_row1[k + pos0]), b0, c1);
        c1 = RINX_FMA(RINX_BROADCAST(A_row1[k + pos1]), b1, c1);
    }
    
    // Store
    int ldc = N;
    RINX_STORE_UNALIGNED(C + (m_start + 0) * ldc + n_start, c0);
    RINX_STORE_UNALIGNED(C + (m_start + 1) * ldc + n_start, c1);
}

// ============================================================================
// GEMM COMPLETO CON TILING (NAIN INTERFACE)
// ============================================================================

// GEMM básico con tiling
void rinx_gemm_basic(const rinx_gemm_args_t* args) {
    int M = args->M, N = args->N, K = args->K;
    const float* A = args->A;
    const float* B = args->B;
    float* C = args->C;
    int lda = args->lda, ldb = args->ldb, ldc = args->ldc;
    
    // Zero C si beta == 0
    if (args->beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            memset(C + i * ldc, 0, N * sizeof(float));
        }
    }
    
    // Tiling configuration
    const rinx_tile_config_t* tiles = &RINX_TILE_CONFIG_SKYLAKE;
    int Mt = tiles->Mt, Nt = tiles->Nt, Kt = tiles->Kt;
    
    // Tiled loops
    for (int mt = 0; mt < M; mt += Mt) {
        int Mt_actual = (mt + Mt < M) ? Mt : (M - mt);
        
        for (int nt = 0; nt < N; nt += Nt) {
            int Nt_actual = (nt + Nt < N) ? Nt : (N - nt);
            
            // Tile de C
            float* C_tile = C + mt * ldc + nt;
            
            for (int kt = 0; kt < K; kt += Kt) {
                int Kt_actual = (kt + Kt < K) ? Kt : (K - kt);
                
                const float* A_tile = A + mt * lda + kt;
                const float* B_tile = B + kt * ldb + nt;
                
                // Micro-kernels según epílogo
                for (int m = 0; m < Mt_actual; m += 8) {
                    int Mw = (m + 8 < Mt_actual) ? 8 : (Mt_actual - m);
                    
                    for (int n = 0; n < Nt_actual; n += 16) {
                        int Nw = (n + 16 < Nt_actual) ? 16 : (Nt_actual - n);
                        
                        if (args->epilogue == RINX_EPILOGUE_NONE) {
                            rinx_gemm_microkernel_8x16(
                                A_tile + m * lda,
                                B_tile + n,
                                C_tile + m * ldc + n,
                                Kt_actual, ldc
                            );
                        } else if (args->epilogue == RINX_EPILOGUE_BIAS_RELU && args->bias) {
                            rinx_gemm_microkernel_8x16_bias_relu(
                                A_tile + m * lda,
                                B_tile + n,
                                args->bias + nt + n,
                                C_tile + m * ldc + n,
                                Kt_actual, ldc, Nw
                            );
                        }
                    }
                }
            }
        }
    }
}

// GEMM con 2:4 sparsity
void rinx_gemm_sparse_2_4(const rinx_gemm_args_t* args, 
                          const rinx_sparse_2_4_matrix_t* sparse_B) {
    int M = args->M, N = args->N, K = args->K;
    const float* A = args->A;
    float* C = args->C;
    int ldc = args->ldc;
    
    // Zero C
    for (int i = 0; i < M; i++) {
        memset(C + i * ldc, 0, N * sizeof(float));
    }
    
    // Sparse GEMM: solo iteramos por non-zeros (mitad de trabajo)
    for (int mt = 0; mt < M; mt += 8) {
        for (int nt = 0; nt < N; nt += 16) {
            rinx_gemm_sparse_2_4_microkernel(
                A, sparse_B, C,
                M, N, K,
                mt, nt
            );
        }
    }
}

// ============================================================================
// CONV2D FUSIONADA (GEMM-based con im2col optimizado)
// ============================================================================

void rinx_conv2d_fused(const float* __restrict input,
                         const float* __restrict weight,
                         const float* __restrict bias,
                         float* __restrict output,
                         const rinx_conv_args_t* args,
                         int batch, int in_h, int in_w) {
    int out_c = args->out_channels;
    int in_c = args->in_channels;
    int k_h = args->kernel_h, k_w = args->kernel_w;
    int stride_h = args->stride_h, stride_w = args->stride_w;
    int pad_h = args->pad_h, pad_w = args->pad_w;
    
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    
    // Optimización: im2col + GEMM fusionado
    // Sin materializar im2col completo (memory-efficient)
    
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow += 16) {
                __m512 acc = RINX_ZERO();
                
                // Convolución como GEMV parcial
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < k_h; kh++) {
                        for (int kw = 0; kw < k_w; kw++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw_base = ow * stride_w - pad_w + kw;
                            
                            if (ih >= 0 && ih < in_h) {
                                // Cargar 16 elementos de input
                                __m512 inp;
                                if (iw_base >= 0 && iw_base + 16 < in_w) {
                                    inp = RINX_LOAD_UNALIGNED(&input[ic * in_h * in_w + ih * in_w + iw_base]);
                                } else {
                                    // Handle padding con mask
                                    inp = RINX_ZERO();  // Simplificado
                                }
                                
                                float w = weight[oc * in_c * k_h * k_w + ic * k_h * k_w + kh * k_w + kw];
                                acc = RINX_FMA(RINX_BROADCAST(w), inp, acc);
                            }
                        }
                    }
                }
                
                // Epílogo fusionado
                if (bias) {
                    __m512 b = RINX_LOAD_UNALIGNED(&bias[oc]);
                    acc = RINX_FMA(acc, RINX_BROADCAST(1.0f), b);
                }
                
                if (args->epilogue & RINX_EPILOGUE_RELU) {
                    acc = rinx_relu_ps(acc);
                } else if (args->epilogue & RINX_EPILOGUE_GELU) {
                    acc = rinx_gelu_ps(acc);
                }
                
                // Store
                int ow_remain = out_w - ow;
                if (ow_remain >= 16) {
                    RINX_STORE_UNALIGNED(&output[oc * out_h * out_w + oh * out_w + ow], acc);
                } else {
                    __mmask16 mask = (1 << ow_remain) - 1;
                    RINX_MASK_STORE(&output[oc * out_h * out_w + oh * out_w + ow], mask, acc);
                }
            }
        }
    }
}

// ============================================================================
// SSM / MAMBA KERNELS
// ============================================================================

// Selective SSM forward (simplified)
void rinx_ssm_selective_forward(const float* __restrict u,  // input
                                   const float* __restrict delta, // time step
                                   const float* __restrict A,     // state matrix
                                   const float* __restrict B,     // input matrix
                                   const float* __restrict C,     // output matrix
                                   float* __restrict y,       // output
                                   float* __restrict h,       // state (temporal)
                                   int batch, int seq_len, int state_dim) {
    // h_t = Ā_t * h_{t-1} + B̄_t * u_t
    // y_t = C_t * h_t
    // Donde Ā = exp(delta * A), B̄ = delta * B
    
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            // Actualizar estado
            for (int d = 0; d < state_dim; d++) {
                float delta_t = delta[b * seq_len + t];
                float A_bar = expf(delta_t * A[d]);  // Discretización ZOH
                float B_bar = delta_t * B[b * seq_len * state_dim + t * state_dim + d];
                
                float u_t = u[b * seq_len + t];
                float h_prev = (t > 0) ? h[b * seq_len * state_dim + (t-1) * state_dim + d] : 0.0f;
                
                h[b * seq_len * state_dim + t * state_dim + d] = A_bar * h_prev + B_bar * u_t;
            }
            
            // Producir output
            float y_t = 0.0f;
            for (int d = 0; d < state_dim; d++) {
                y_t += C[b * seq_len * state_dim + t * state_dim + d] * 
                       h[b * seq_len * state_dim + t * state_dim + d];
            }
            y[b * seq_len + t] = y_t;
        }
    }
}

// Parallel scan para SSM (associative scan)
// Más eficiente en GPU, implementación CPU con tiling
void rinx_parallel_scan_ssm(const float* __restrict inputs,
                             float* __restrict outputs,
                             float* __restrict states,
                             int len, int state_dim) {
    // Up-sweep (reduce)
    for (int d = 0; d < state_dim; d++) {
        for (int stride = 1; stride < len; stride *= 2) {
            for (int i = 0; i < len; i += 2 * stride) {
                int idx = i + 2 * stride - 1;
                if (idx < len) {
                    // Combinar: states[idx] = combine(states[idx-stride], states[idx])
                    states[idx * state_dim + d] += states[(idx - stride) * state_dim + d];
                }
            }
        }
    }
    
    // Down-sweep (broadcast)
    for (int d = 0; d < state_dim; d++) {
        for (int stride = len / 2; stride >= 1; stride /= 2) {
            for (int i = stride; i < len; i += 2 * stride) {
                if (i + stride < len) {
                    states[(i + stride) * state_dim + d] += states[i * state_dim + d];
                }
            }
        }
    }
}

// ============================================================================
// BENCHMARK Y UTILIDADES
// ============================================================================

#include <time.h>

static inline double rinx_get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    double time_ms;
    double gflops;
    double throughput_gbps;
    const char* kernel_name;
} rinx_benchmark_result_t;

rinx_benchmark_result_t rinx_benchmark_gemm(int M, int N, int K, int num_runs) {
    // Allocar matrices
    float* A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, M * N * sizeof(float));
    float* bias = (float*)aligned_alloc(64, N * sizeof(float));
    
    // Inicializar
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N; i++) bias[i] = (float)rand() / RAND_MAX;
    
    rinx_gemm_args_t args = {
        .M = M, .N = N, .K = K,
        .alpha = 1.0f, .beta = 0.0f,
        .A = A, .lda = K,
        .B = B, .ldb = N,
        .C = C, .ldc = N,
        .bias = bias,
        .epilogue = RINX_EPILOGUE_BIAS_RELU,
        .sparsity_A = RINX_SPARSE_NONE,
        .sparsity_B = RINX_SPARSE_NONE,
        .algo = RINX_GEMM_ALGO_TILED
    };
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        rinx_gemm_basic(&args);
    }
    
    // Benchmark
    double start = rinx_get_time();
    for (int i = 0; i < num_runs; i++) {
        rinx_gemm_basic(&args);
    }
    double end = rinx_get_time();
    
    double time_ms = (end - start) * 1000.0 / num_runs;
    double flops = 2.0 * M * N * K;  // multiply-add
    double gflops = (flops / time_ms) / 1e6;
    
    free(A); free(B); free(C); free(bias);
    
    return (rinx_benchmark_result_t){
        .time_ms = time_ms,
        .gflops = gflops,
        .throughput_gbps = 0,
        .kernel_name = "rinx_gemm_basic"
    };
}

// ============================================================================

#endif // RINX_CUTLASS_KERNELS_H
