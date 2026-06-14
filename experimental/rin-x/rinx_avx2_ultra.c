/*
 * RIN-X ULTRA-FAST AVX2 - Vectorización Completa
 * Objetivo: < 0.01 ms por inference (superar 3.5× vs ONNX)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

// ============================================================================
// CONFIGURACIÓN PARA MÁXIMA VELOCIDAD
// ============================================================================

#define IN_DIM 784
#define H1 128
#define H2 128
#define OUT_DIM 10
#define TIME_STEPS 3

#define THRESHOLD 0.5f
#define DECAY 0.8f
#define SCALE 0.01f  // Quantization scale

// Alinear dimensiones para SIMD
#define H1_ALIGN ((H1 + 7) & ~7)      // Múltiplo de 8 (AVX2)
#define H2_ALIGN ((H2 + 7) & ~7)
#define OUT_ALIGN ((OUT_DIM + 7) & ~7)

// ============================================================================
// INT8 GEMV ULTRA-VECTORIZADO AVX2
// ============================================================================

/*
 * GEMV INT8 con desempaquetado a int16 y madd (multiply-add)
 * Procesa 32 elementos por iteración (256 bits / 8 bits)
 * Acumulación en int32 para evitar overflow
 */
static inline void int8_gemv_avx2_optimized(const int8_t* __restrict A,
                                             const float* __restrict x,
                                             float* __restrict y,
                                             int rows, int cols) {
    // Procesar 4 filas a la vez para mejor ILP
    for (int i = 0; i < rows; i += 4) {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        
        // Procesar columnas en bloques de 32 (256 bits / 8 bits)
        for (int j = 0; j < cols; j += 32) {
            // Prefetch siguiente bloque
            _mm_prefetch((const char*)(A + (i+4) * cols + j), _MM_HINT_T0);
            _mm_prefetch((const char*)(x + j + 32), _MM_HINT_T0);
            
            // Cargar 32 elementos de x (int8) y extender a int16
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + j));
            
            // Desempaquetar a int16: low y high halves
            __m256i xv_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 0));
            __m256i xv_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
            
            // Para cada una de las 4 filas
            for (int r = 0; r < 4 && (i + r) < rows; r++) {
                // Cargar 32 pesos de esta fila
                __m256i av = _mm256_loadu_si256((__m256i*)(A + (i + r) * cols + j));
                __m256i av_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 0));
                __m256i av_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
                
                // Multiply-add int16 → int32
                // _mm256_madd_epi16: multiplica pares y suma
                __m256i prod_lo = _mm256_madd_epi16(av_lo, xv_lo);
                __m256i prod_hi = _mm256_madd_epi16(av_hi, xv_hi);
                
                // Seleccionar acumulador correcto
                __m256i* acc = (r == 0) ? &acc0 : (r == 1) ? &acc1 : (r == 2) ? &acc2 : &acc3;
                *acc = _mm256_add_epi32(*acc, prod_lo);
                *acc = _mm256_add_epi32(*acc, prod_hi);
            }
        }
        
        // Reducir y convertir a float
        for (int r = 0; r < 4 && (i + r) < rows; r++) {
            __m256i acc = (r == 0) ? acc0 : (r == 1) ? acc1 : (r == 2) ? acc2 : acc3;
            
            // Reducción horizontal AVX2
            __m128i acc_lo = _mm256_extracti128_si256(acc, 0);
            __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum = _mm_add_epi32(acc_lo, acc_hi);
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            
            int32_t int_sum = _mm_extract_epi32(sum, 0);
            y[i + r] = (float)int_sum * SCALE;
        }
    }
}

// ============================================================================
// LIF FORWARD VECTORIZADO
// ============================================================================

/*
 * LIF con AVX2 para las operaciones vectoriales
 * v_mem = v_mem * decay + current
 * spike = v_mem >= threshold ? 1 : 0
 * v_mem = v_mem * (1 - spike)  // reset
 */
static inline void lif_forward_avx2(const int8_t* __restrict w,
                                     const float* __restrict x,
                                     float* __restrict out,
                                     int rows, int cols,
                                     float* __restrict v_mem,
                                     int time_steps) {
    // Inicializar v_mem a cero
    memset(v_mem, 0, rows * sizeof(float));
    
    // Pre-cargar constantes en registers
    __m256 decay_vec = _mm256_set1_ps(DECAY);
    __m256 threshold_vec = _mm256_set1_ps(THRESHOLD);
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 zero_vec = _mm256_setzero_ps();
    
    // Buffer para current
    float current[rows];
    
    for (int t = 0; t < time_steps; t++) {
        // Compute current (GEMV)
        int8_gemv_avx2_optimized(w, x, current, rows, cols);
        
        // LIF dynamics vectorizado
        int i = 0;
        for (; i <= rows - 8; i += 8) {
            // Cargar v_mem
            __m256 v = _mm256_loadu_ps(v_mem + i);
            
            // Cargar current
            __m256 c = _mm256_loadu_ps(current + i);
            
            // v = v * decay + current
            v = _mm256_fmadd_ps(v, decay_vec, c);
            
            // spike = v >= threshold ? 1 : 0
            __m256 spike_mask = _mm256_cmp_ps(v, threshold_vec, _CMP_GE_OQ);
            __m256 spike = _mm256_and_ps(spike_mask, one_vec);
            
            // v = v * (1 - spike)  // reset where spike
            __m256 reset = _mm256_sub_ps(one_vec, spike);
            v = _mm256_mul_ps(v, reset);
            
            // Store v_mem
            _mm256_storeu_ps(v_mem + i, v);
        }
        
        // Resto escalar
        for (; i < rows; i++) {
            v_mem[i] = v_mem[i] * DECAY + current[i];
            if (v_mem[i] >= THRESHOLD) {
                v_mem[i] = 0.0f;
            }
        }
    }
    
    // Promedio sobre time steps y store output
    float inv_ts = 1.0f / time_steps;
    __m256 inv_ts_vec = _mm256_set1_ps(inv_ts);
    
    int i = 0;
    for (; i <= rows - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(v_mem + i);
        v = _mm256_mul_ps(v, inv_ts_vec);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < rows; i++) {
        out[i] = v_mem[i] * inv_ts;
    }
}

// ============================================================================
// MODELO COMPLETO AVX2
// ============================================================================

typedef struct {
    int8_t w1[H1 * IN_DIM];
    int8_t w2[H2 * H1];
    int8_t w3[OUT_DIM * H2];
} rinx_avx2_model_t;

void rinx_avx2_inference(const rinx_avx2_model_t* __restrict model,
                          const float* __restrict input,
                          float* __restrict output) {
    // Buffers en stack (L1 cache)
    float h1[H1_ALIGN] __attribute__((aligned(32)));
    float h2[H2_ALIGN] __attribute__((aligned(32)));
    float v1[H1_ALIGN] __attribute__((aligned(32)));
    float v2[H2_ALIGN] __attribute__((aligned(32)));
    
    // Capa 1: Input → Hidden1
    lif_forward_avx2(model->w1, input, h1, H1, IN_DIM, v1, TIME_STEPS);
    
    // Capa 2: Hidden1 → Hidden2
    lif_forward_avx2(model->w2, h1, h2, H2, H1, v2, TIME_STEPS);
    
    // Capa 3: Hidden2 → Output (sin LIF, solo linear)
    // Usar versión scalar para las 10 salidas (no vale la pena vectorizar)
    for (int i = 0; i < OUT_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < H2; j++) {
            sum += (float)model->w3[i * H2 + j] * h2[j];
        }
        output[i] = sum * SCALE;
    }
}

// ============================================================================
// INICIALIZACIÓN
// ============================================================================

void rinx_avx2_init_random(rinx_avx2_model_t* model) {
    srand(42);
    for (int i = 0; i < H1 * IN_DIM; i++) {
        model->w1[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < H2 * H1; i++) {
        model->w2[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < OUT_DIM * H2; i++) {
        model->w3[i] = (int8_t)(rand() % 256 - 128);
    }
}

// ============================================================================
// BENCHMARK
// ============================================================================

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=====================================\n");
    printf("RIN-X AVX2 ULTRA-FAST\n");
    printf("=====================================\n\n");
    
    // Verificar AVX2 disponible
    #ifdef __AVX2__
    printf("✓ Compilado con AVX2\n\n");
    #else
    printf("⚠️ AVX2 no disponible en compilación\n\n");
    #endif
    
    // Crear modelo
    rinx_avx2_model_t model;
    rinx_avx2_init_random(&model);
    
    // Input alineado
    float input[IN_DIM] __attribute__((aligned(32)));
    for (int i = 0; i < IN_DIM; i++) {
        input[i] = (float)(rand() % 20 - 10) / 100.0f;
    }
    
    float output[OUT_ALIGN];
    
    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < 1000; i++) {
        rinx_avx2_inference(&model, input, output);
    }
    
    // Benchmark preciso
    printf("Benchmarking...\n\n");
    
    int runs = 50000;
    double start = get_time();
    for (int i = 0; i < runs; i++) {
        rinx_avx2_inference(&model, input, output);
    }
    double end = get_time();
    
    double time_ms = (end - start) * 1000.0 / runs;
    double throughput = 1000.0 / time_ms;
    
    printf("RESULTADOS:\n");
    printf("  Time per inference: %.4f ms\n", time_ms);
    printf("  Throughput: %.0f inf/s\n", throughput);
    printf("  Cycles (est. 3GHz): ~%.0f\n", time_ms * 3e6);
    printf("\n");
    
    // Comparación
    double onnx_time = 0.035;  // Medido anteriormente
    double speedup = onnx_time / time_ms;
    
    printf("COMPARACIÓN:\n");
    printf("  ONNX Runtime: %.3f ms\n", onnx_time);
    printf("  RIN-X AVX2:   %.4f ms\n", time_ms);
    printf("\n");
    
    if (speedup > 1.0) {
        printf("  ✅ RIN-X es %.2f× MÁS RÁPIDO que ONNX!\n", speedup);
        if (speedup > 3.0) {
            printf("  🚀 Superamos objetivo de 3×!\n");
        }
    } else {
        printf("  ⚠️ RIN-X es %.2f× más lento\n", 1.0/speedup);
    }
    
    printf("\nOptimizaciones aplicadas:\n");
    printf("  ✓ AVX2 full vectorization\n");
    printf("  ✓ _mm256_madd_epi16 para GEMV\n");
    printf("  ✓ 4-way unrolling en filas\n");
    printf("  ✓ Aligned memory (32 bytes)\n");
    printf("  ✓ Prefetching agresivo\n");
    printf("  ✓ Stack-only (no heap)\n");
    
    printf("\n=====================================\n");
    printf("Benchmark completado.\n");
    printf("=====================================\n");
    
    return 0;
}
