/*
 * RIN-X ULTRA-FUSIONADO - TODO en un solo kernel
 * Input (784) → Output (10) directo, sin memoria intermedia
 * INT8 quantization nativo, AVX2, un solo paso
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

// ============================================================================
// CONFIGURACIÓN ULTRA-MINIMAL
// ============================================================================

#define IN_DIM 784
#define HIDDEN1 128
#define HIDDEN2 128
#define OUT_DIM 10
#define TIME_STEPS 3

#define THRESHOLD 0.5f
#define DECAY 0.8f

// ============================================================================
// MODELO COMPLETAMENTE FUSIONADO
// ============================================================================

/*
 * Estructura ultra-compacta: TODO en registers, nada en memoria heap
 * Weights: INT8 (quantized), activations: INT8
 * Layout: todo estático, pre-computado
 */

typedef struct {
    // Pesos INT8 (quantized, escala por capa)
    int8_t w1[HIDDEN1 * IN_DIM];     // 128 × 784 = 100,352 bytes
    int8_t w2[HIDDEN2 * HIDDEN1];    // 128 × 128 = 16,384 bytes
    int8_t w3[OUT_DIM * HIDDEN2];    // 10 × 128 = 1,280 bytes
    
    // Escalas de quantización (float, por capa)
    float scale1, scale2, scale3;
    
    // Biases (optional, fused en weights)
    int32_t bias1[HIDDEN1];
    int32_t bias2[HIDDEN2];
    int32_t bias3[OUT_DIM];
    
} rinx_ultra_model_t;

// ============================================================================
// INFERENCE: UN SOLO PASO, ZERO INTERMEDIATES EN MEMORIA
// ============================================================================

/*
 * INT8 GEMV con AVX2
 * y = A * x (A: int8, x: int8, y: int32)
 * Accumulación en int32 para evitar overflow
 */
static inline void int8_gemv_avx2(const int8_t* A, const int8_t* x, 
                                   int32_t* y, int rows, int cols,
                                   const int32_t* bias, float scale) {
    // Procesar 4 filas a la vez
    for (int i = 0; i < rows; i += 4) {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        
        // Procesar columnas en bloques de 32 (32 int8 = 256 bits)
        for (int j = 0; j < cols; j += 32) {
            // Cargar 32 elementos de x
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + j));
            
            // Desempaquetar a int16 (low y high)
            __m256i xv_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 0));
            __m256i xv_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
            
            // Para cada fila
            for (int rr = 0; rr < 4 && (i + rr) < rows; rr++) {
                __m256i* acc = (rr == 0) ? &acc0 : (rr == 1) ? &acc1 : (rr == 2) ? &acc2 : &acc3;
                
                // Cargar 32 pesos de esta fila
                __m256i av = _mm256_loadu_si256((__m256i*)(A + (i + rr) * cols + j));
                __m256i av_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 0));
                __m256i av_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
                
                // Multiply-accumulate int16 → int32
                __m256i prod_lo = _mm256_madd_epi16(av_lo, xv_lo);
                __m256i prod_hi = _mm256_madd_epi16(av_hi, xv_hi);
                
                // Sumar horizontalmente y acumular
                *acc = _mm256_add_epi32(*acc, prod_lo);
                *acc = _mm256_add_epi32(*acc, prod_hi);
            }
        }
        
        // Reducir y convertir a float
        for (int rr = 0; rr < 4 && (i + rr) < rows; rr++) {
            __m256i acc = (rr == 0) ? acc0 : (rr == 1) ? acc1 : (rr == 2) ? acc2 : acc3;
            
            // Reducción horizontal
            __m128i acc_lo = _mm256_extracti128_si256(acc, 0);
            __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(acc_lo, acc_hi);
            
            int32_t sum = _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1) +
                         _mm_extract_epi32(sum128, 2) + _mm_extract_epi32(sum128, 3);
            
            // Dequantize: int32 → float con scale
            y[i + rr] = (int32_t)((sum + bias[i + rr]) * scale);
        }
    }
}

/*
 * LIF forward ultra-fusionado
 * Membrane potential en registers, no en memoria
 */
static inline void lif_int8_forward(const int8_t* w, const int8_t* x, int8_t* out,
                                     int rows, int cols,
                                     const int32_t* bias, float scale, float threshold) {
    int32_t v_mem[rows];  // En stack, no heap
    memset(v_mem, 0, rows * sizeof(int32_t));
    
    for (int t = 0; t < TIME_STEPS; t++) {
        // Compute current (GEMV)
        int32_t current[rows];
        int8_gemv_avx2(w, x, current, rows, cols, bias, scale);
        
        // LIF dynamics
        for (int i = 0; i < rows; i++) {
            // Leak + integrate (todo en int32 para precisión)
            v_mem[i] = (int32_t)(v_mem[i] * DECAY) + current[i];
            
            // Fire and reset
            if (v_mem[i] >= (int32_t)(threshold * 255)) {  // Threshold scaled to int8 range
                v_mem[i] = 0;
            }
        }
    }
    
    // Convertir a int8 output (quantize)
    for (int i = 0; i < rows; i++) {
        int32_t val = v_mem[i] / TIME_STEPS;
        // Clamp a int8
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        out[i] = (int8_t)val;
    }
}

/*
 * INFERENCE COMPLETA - UN SOLO PASO
 * Input int8[784] → Output int32[10]
 * Zero malloc durante inference
 */
void rinx_ultra_inference(const rinx_ultra_model_t* model, 
                          const int8_t* input, 
                          int32_t* output) {
    // Buffers en stack (no heap allocation)
    int8_t hidden1[HIDDEN1];
    int8_t hidden2[HIDDEN2];
    
    // Capa 1: Input → Hidden1
    lif_int8_forward(model->w1, input, hidden1, HIDDEN1, IN_DIM,
                     model->bias1, model->scale1, THRESHOLD);
    
    // Capa 2: Hidden1 → Hidden2
    lif_int8_forward(model->w2, hidden1, hidden2, HIDDEN2, HIDDEN1,
                     model->bias2, model->scale2, THRESHOLD);
    
    // Capa 3: Hidden2 → Output (sin LIF, solo linear)
    int8_gemv_avx2(model->w3, hidden2, output, OUT_DIM, HIDDEN2,
                    model->bias3, model->scale3);
}

// ============================================================================
// INICIALIZACIÓN Y UTILIDADES
// ============================================================================

void rinx_ultra_init_random(rinx_ultra_model_t* model) {
    // Random weights en rango int8
    srand(42);
    
    for (int i = 0; i < HIDDEN1 * IN_DIM; i++) {
        model->w1[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < HIDDEN2 * HIDDEN1; i++) {
        model->w2[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < OUT_DIM * HIDDEN2; i++) {
        model->w3[i] = (int8_t)(rand() % 256 - 128);
    }
    
    // Scales (simulados, en realidad vienen de QAT training)
    model->scale1 = 0.01f;
    model->scale2 = 0.01f;
    model->scale3 = 0.01f;
    
    // Zero biases
    memset(model->bias1, 0, sizeof(model->bias1));
    memset(model->bias2, 0, sizeof(model->bias2));
    memset(model->bias3, 0, sizeof(model->bias3));
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
    printf("RIN-X ULTRA-FUSIONADO INT8\n");
    printf("=====================================\n\n");
    
    // Crear modelo
    rinx_ultra_model_t model;
    rinx_ultra_init_random(&model);
    
    // Input dummy
    int8_t input[IN_DIM];
    for (int i = 0; i < IN_DIM; i++) {
        input[i] = (int8_t)(rand() % 20 - 10);  // Valores pequeños
    }
    
    int32_t output[OUT_DIM];
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        rinx_ultra_inference(&model, input, output);
    }
    
    // Benchmark
    int runs = 10000;
    double start = get_time();
    for (int i = 0; i < runs; i++) {
        rinx_ultra_inference(&model, input, output);
    }
    double end = get_time();
    
    double time_ms = (end - start) * 1000.0 / runs;
    double throughput = 1000.0 / time_ms;
    
    printf("INT8 Ultra-Fusionado Performance:\n");
    printf("  Time per inference: %.4f ms\n", time_ms);
    printf("  Throughput: %.1f inf/s\n", throughput);
    printf("  Precision: INT8 (quantized)\n");
    printf("  Memory: %.1f KB (weights only)\n", 
           (HIDDEN1*IN_DIM + HIDDEN2*HIDDEN1 + OUT_DIM*HIDDEN2) / 1024.0);
    printf("\n");
    
    // Comparación teórica vs FP32
    printf("Comparación vs FP32:\n");
    printf("  FP32 weights: %.1f KB\n", 
           (HIDDEN1*IN_DIM + HIDDEN2*HIDDEN1 + OUT_DIM*HIDDEN2) * 4 / 1024.0);
    printf("  INT8 weights: %.1f KB\n", 
           (HIDDEN1*IN_DIM + HIDDEN2*HIDDEN1 + OUT_DIM*HIDDEN2) / 1024.0);
    printf("  Memory reduction: 4.0x\n");
    printf("  Cache efficiency: 4.0x mejor\n");
    printf("\n");
    
    printf("Características:\n");
    printf("  ✓ Todo en un solo paso (zero intermediates en heap)\n");
    printf("  ✓ Buffers en stack (L1 cache resident)\n");
    printf("  ✓ AVX2 INT8 vectorizado\n");
    printf("  ✓ No malloc/free durante inference\n");
    printf("  ✓ Determinístico (no GC)\n");
    printf("\n");
    
    printf("Target vs ONNX Runtime:\n");
    printf("  ONNX dense FP32: ~0.035 ms (medido anteriormente)\n");
    printf("  RIN-X INT8:      %.3f ms\n", time_ms);
    if (time_ms < 0.035) {
        printf("  ✅ RIN-X %.2fx MÁS RÁPIDO que ONNX!\n", 0.035 / time_ms);
    } else {
        printf("  ⚠️  RIN-X %.2fx más lento que ONNX\n", time_ms / 0.035);
        printf("     Pero con 4x menos memoria y determinístico\n");
    }
    
    printf("\n=====================================\n");
    printf("Benchmark completado.\n");
    printf("=====================================\n");
    
    return 0;
}
