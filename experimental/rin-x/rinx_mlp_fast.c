/*
 * RIN-X MLP FAST - Sin SNN time steps, solo ReLU
 * Más simple = más rápido. Target: 3-4× vs ONNX
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

#define IN_DIM 784
#define H1 256   // Más grande para más trabajo por inference
#define H2 256
#define OUT_DIM 10

#define SCALE 0.01f

// Modelo
typedef struct {
    int8_t w1[H1 * IN_DIM] __attribute__((aligned(32)));
    int8_t w2[H2 * H1] __attribute__((aligned(32)));
    int8_t w3[OUT_DIM * H2] __attribute__((aligned(32)));
} mlp_model_t;

// ReLU vectorizado AVX2
static inline __m256 relu_ps(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

// GEMV + ReLU fusionado
static void gemv_relu_avx2(const int8_t* __restrict A,
                            const float* __restrict x,
                            float* __restrict y,
                            int rows, int cols) {
    
    for (int i = 0; i < rows; i += 8) {
        __m256i acc[8];
        for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_si256();
        
        for (int j = 0; j < cols; j += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + j));
            __m256i xv_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 0));
            __m256i xv_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
            
            for (int r = 0; r < 8 && (i + r) < rows; r++) {
                __m256i av = _mm256_loadu_si256((__m256i*)(A + (i + r) * cols + j));
                __m256i av_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 0));
                __m256i av_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(av_lo, xv_lo));
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(av_hi, xv_hi));
            }
        }
        
        // Reducir, escalar, aplicar ReLU
        for (int r = 0; r < 8 && (i + r) < rows; r++) {
            __m128i lo = _mm256_extracti128_si256(acc[r], 0);
            __m128i hi = _mm256_extracti128_si256(acc[r], 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            float val = (float)_mm_extract_epi32(sum, 0) * SCALE;
            y[i + r] = val > 0 ? val : 0;  // ReLU
        }
    }
}

// MLP inference: Input -> ReLU -> Hidden -> ReLU -> Output
void mlp_inference(const mlp_model_t* __restrict model,
                     const float* __restrict input,
                     float* __restrict output) {
    
    float h1[H1] __attribute__((aligned(32)));
    float h2[H2] __attribute__((aligned(32)));
    
    // Capa 1: GEMV + ReLU
    gemv_relu_avx2(model->w1, input, h1, H1, IN_DIM);
    
    // Capa 2: GEMV + ReLU
    // Convertir h1 a int8 para siguiente capa
    int8_t h1_int8[H1];
    for (int i = 0; i < H1; i++) {
        int32_t val = (int32_t)(h1[i] * 100);
        if (val > 127) val = 127;
        h1_int8[i] = (int8_t)val;
    }
    gemv_relu_avx2(model->w2, h1, h2, H2, H1);
    
    // Capa 3: GEMV (sin ReLU, es output)
    int32_t acc[OUT_DIM] = {0};
    for (int i = 0; i < OUT_DIM; i++) {
        for (int j = 0; j < H2; j++) {
            acc[i] += (int32_t)model->w3[i * H2 + j] * (int32_t)(h2[j] * 100);
        }
        output[i] = (float)acc[i] * SCALE / 100.0f;
    }
}

// Init
void init_mlp(mlp_model_t* m) {
    srand(42);
    for (int i = 0; i < H1 * IN_DIM; i++) m->w1[i] = rand() % 256 - 128;
    for (int i = 0; i < H2 * H1; i++) m->w2[i] = rand() % 256 - 128;
    for (int i = 0; i < OUT_DIM * H2; i++) m->w3[i] = rand() % 256 - 128;
}

// Timer
static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=====================================\n");
    printf("RIN-X MLP FAST (No SNN)\n");
    printf("Arquitectura: 784→256→256→10, ReLU\n");
    printf("=====================================\n\n");
    
    mlp_model_t model;
    init_mlp(&model);
    
    float input[IN_DIM] __attribute__((aligned(32)));
    float output[OUT_DIM];
    
    for (int i = 0; i < IN_DIM; i++) {
        input[i] = (float)(rand() % 20 - 10) / 100.0f;
    }
    
    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < 1000; i++) {
        mlp_inference(&model, input, output);
    }
    
    // Benchmark
    int runs = 100000;
    double start = get_time();
    for (int i = 0; i < runs; i++) {
        mlp_inference(&model, input, output);
    }
    double end = get_time();
    
    double time_ms = (end - start) * 1000.0 / runs;
    double throughput = 1000.0 / time_ms;
    
    printf("RESULTADOS MLP:\n");
    printf("  Time: %.5f ms\n", time_ms);
    printf("  Throughput: %.0f inf/s\n", throughput);
    printf("  Ops: %d (2× más que SNN 128)\n", 2 * (784*256 + 256*256 + 256*10));
    printf("\n");
    
    // Comparación
    double onnx_mlp = 0.050;  // Estimado para MLP 256
    double speedup = onnx_mlp / time_ms;
    printf("COMPARACIÓN (estimada):\n");
    printf("  ONNX MLP 256: ~%.3f ms (estimado)\n", onnx_mlp);
    printf("  RIN-X MLP:    %.5f ms\n", time_ms);
    printf("  Speedup:      %.2f×\n", speedup);
    printf("\n");
    
    if (speedup >= 3.0) {
        printf("🎉 3×+ LOGRADO con MLP!\n");
    } else {
        printf("ℹ️ MLP más grande = más trabajo, pero throughput alto\n");
    }
    
    printf("\n=====================================\n");
    
    return 0;
}
