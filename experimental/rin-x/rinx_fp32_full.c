/*
 * RIN-X FP32 FULL - Sin cuantización intermedia
 * Todo en FP32 para máxima velocidad
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

#define IN_DIM 784
#define H1 256
#define H2 256
#define OUT_DIM 10

// Modelo FP32 completo
typedef struct {
    float w1[H1 * IN_DIM] __attribute__((aligned(32)));
    float b1[H1];
    float w2[H2 * H1] __attribute__((aligned(32)));
    float b2[H2];
    float w3[OUT_DIM * H2] __attribute__((aligned(32)));
    float b3[OUT_DIM];
} mlp_fp32_model_t;

// ReLU AVX2
static inline __m256 relu_ps(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

// GEMV FP32 + Bias + ReLU
static void gemv_fp32_bias_relu(const float* __restrict W,
                                  const float* __restrict x,
                                  const float* __restrict bias,
                                  float* __restrict y,
                                  int rows, int cols) {
    
    for (int i = 0; i < rows; i++) {
        __m256 sum = _mm256_setzero_ps();
        
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 wv = _mm256_loadu_ps(W + i * cols + j);
            __m256 xv = _mm256_loadu_ps(x + j);
            sum = _mm256_fmadd_ps(wv, xv, sum);
        }
        
        // Reducir suma
        float result = bias[i];
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        for (int k = 0; k < 8; k++) result += temp[k];
        
        // Resto scalar
        for (; j < cols; j++) {
            result += W[i * cols + j] * x[j];
        }
        
        y[i] = result > 0 ? result : 0;  // ReLU
    }
}

// GEMV FP32 + Bias (final)
static void gemv_fp32_bias_final(const float* __restrict W,
                                   const float* __restrict x,
                                   const float* __restrict bias,
                                   float* __restrict y,
                                   int rows, int cols) {
    
    for (int i = 0; i < rows; i++) {
        float sum = bias[i];
        
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        
        y[i] = sum;
    }
}

// Inference FP32
void mlp_fp32_inference(const mlp_fp32_model_t* __restrict model,
                          const float* __restrict input,
                          float* __restrict output) {
    
    float h1[H1];
    float h2[H2];
    
    // Capa 1: GEMV + Bias + ReLU
    gemv_fp32_bias_relu(model->w1, input, model->b1, h1, H1, IN_DIM);
    
    // Capa 2: GEMV + Bias + ReLU
    gemv_fp32_bias_relu(model->w2, h1, model->b2, h2, H2, H1);
    
    // Capa 3: GEMV + Bias
    gemv_fp32_bias_final(model->w3, h2, model->b3, output, OUT_DIM, H2);
}

// Init
void init_fp32_model(mlp_fp32_model_t* m) {
    srand(42);
    
    for (int i = 0; i < H1 * IN_DIM; i++) m->w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < H2 * H1; i++) m->w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < OUT_DIM * H2; i++) m->w3[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    for (int i = 0; i < H1; i++) m->b1[i] = 0.0f;
    for (int i = 0; i < H2; i++) m->b2[i] = 0.0f;
    for (int i = 0; i < OUT_DIM; i++) m->b3[i] = 0.0f;
}

// Timer
static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=====================================\n");
    printf("RIN-X FP32 FULL (Sin Cuantización)\n");
    printf("=====================================\n\n");
    
    mlp_fp32_model_t model;
    init_fp32_model(&model);
    
    float input[IN_DIM];
    for (int i = 0; i < IN_DIM; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
    
    float output[OUT_DIM];
    
    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < 1000; i++) {
        mlp_fp32_inference(&model, input, output);
    }
    
    // Benchmark
    int runs = 100000;
    double start = get_time();
    for (int i = 0; i < runs; i++) {
        mlp_fp32_inference(&model, input, output);
    }
    double end = get_time();
    
    double time_ms = (end - start) * 1000.0 / runs;
    double throughput = 1000.0 / time_ms;
    
    printf("RESULTADOS FP32 FULL:\n");
    printf("  Time: %.5f ms\n", time_ms);
    printf("  Throughput: %.0f inf/s\n", throughput);
    printf("\n");
    
    double onnx = 0.0159;  // Medido real
    double speedup = onnx / time_ms;
    printf("COMPARACIÓN:\n");
    printf("  ONNX (medido): %.4f ms\n", onnx);
    printf("  RIN-X FP32:    %.5f ms\n", time_ms);
    printf("  Speedup:       %.2f×\n", speedup);
    printf("\n");
    
    if (speedup >= 3.0) {
        printf("🎉 3×+ LOGRADO sin cuantización!\n");
    } else if (speedup >= 2.0) {
        printf("✅ 2×+ logrado\n");
    } else if (speedup >= 1.0) {
        printf("ℹ️  Paridad o ligera ventaja\n");
    } else {
        printf("⚠️  Aún más lento que ONNX\n");
    }
    
    printf("\nNotas:\n");
    printf("  - Sin cuantización INT8 (todo FP32)\n");
    printf("  - Sin conversión entre capas\n");
    printf("  - AVX2 con FMA\n");
    printf("  - Más memory bandwidth que INT8\n");
    
    printf("\n=====================================\n");
    
    return 0;
}
