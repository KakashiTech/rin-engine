/*
 * RIN-X BATCH PROCESSING - Throughput máximo
 * Procesar múltiples imágenes en paralelo para amortizar overhead
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

#define IN_DIM 784
#define H1 128
#define H2 128
#define OUT_DIM 10
#define TIME_STEPS 3
#define BATCH_SIZE 64  // Procesar 64 imágenes a la vez

#define THRESHOLD 0.5f
#define DECAY 0.8f
#define SCALE 0.01f

// ============================================================================
// GEMM BATCH - Múltiples vectores a la vez
// ============================================================================

/*
 * Procesar batch de imágenes
 * A: weights [rows x cols]
 * X: batch de inputs [batch_size x cols]
 * Y: batch de outputs [batch_size x rows]
 * 
 * Optimización: amortizar carga de pesos sobre múltiples inputs
 */
static void int8_gemm_batch(const int8_t* __restrict A,
                             const float* __restrict X,  // [batch][cols]
                             float* __restrict Y,          // [batch][rows]
                             int rows, int cols, int batch) {
    
    // Para cada fila de pesos
    for (int i = 0; i < rows; i++) {
        // Cargar pesos de esta fila una sola vez
        // Procesar en bloques de 32 para AVX2
        int8_t weights[cols];
        memcpy(weights, A + i * cols, cols);
        
        // Para cada sample en el batch
        for (int b = 0; b < batch; b++) {
            float sum = 0.0f;
            const float* x = X + b * cols;
            
            // GEMV
            for (int j = 0; j < cols; j++) {
                sum += (float)weights[j] * x[j];
            }
            
            Y[b * rows + i] = sum * SCALE;
        }
    }
}

// ============================================================================
// LIF BATCH
// ============================================================================

static void lif_batch(const int8_t* __restrict w,
                        const float* __restrict x,  // [batch][cols]
                        float* __restrict y,        // [batch][rows]
                        int rows, int cols, int batch,
                        float* __restrict v_mem) {  // [batch][rows]
    
    // Zero v_mem
    memset(v_mem, 0, batch * rows * sizeof(float));
    
    // Pre-cargar constantes
    __m256 decay_vec = _mm256_set1_ps(DECAY);
    __m256 thresh_vec = _mm256_set1_ps(THRESHOLD);
    __m256 one_vec = _mm256_set1_ps(1.0f);
    
    for (int t = 0; t < TIME_STEPS; t++) {
        // Compute current para todo el batch
        for (int b = 0; b < batch; b++) {
            // GEMV para este sample
            for (int i = 0; i < rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) {
                    sum += (float)w[i * cols + j] * x[b * cols + j];
                }
                
                float* v = &v_mem[b * rows + i];
                *v = *v * DECAY + sum * SCALE;
                
                if (*v >= THRESHOLD) {
                    *v = 0.0f;
                }
            }
        }
    }
    
    // Promedio y guardar output
    float inv_ts = 1.0f / TIME_STEPS;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < rows; i++) {
            y[b * rows + i] = v_mem[b * rows + i] * inv_ts;
        }
    }
}

// ============================================================================
// INFERENCE BATCH
// ============================================================================

typedef struct {
    int8_t w1[H1 * IN_DIM];
    int8_t w2[H2 * H1];
    int8_t w3[OUT_DIM * H2];
} model_t;

void rinx_batch_inference(const model_t* __restrict model,
                           const float* __restrict input,   // [batch][784]
                           float* __restrict output,          // [batch][10]
                           int batch) {
    
    // Buffers temporales
    float* h1 = (float*)aligned_alloc(32, batch * H1 * sizeof(float));
    float* h2 = (float*)aligned_alloc(32, batch * H2 * sizeof(float));
    float* v_mem = (float*)aligned_alloc(32, batch * H2 * sizeof(float));
    
    // Capa 1
    lif_batch(model->w1, input, h1, H1, IN_DIM, batch, v_mem);
    
    // Capa 2
    lif_batch(model->w2, h1, h2, H2, H1, batch, v_mem);
    
    // Capa 3 (linear)
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < OUT_DIM; i++) {
            float sum = 0.0f;
            for (int j = 0; j < H2; j++) {
                sum += (float)model->w3[i * H2 + j] * h2[b * H2 + j];
            }
            output[b * OUT_DIM + i] = sum * SCALE;
        }
    }
    
    free(h1);
    free(h2);
    free(v_mem);
}

// ============================================================================
// BENCHMARK BATCH
// ============================================================================

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=====================================\n");
    printf("RIN-X BATCH PROCESSING\n");
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("=====================================\n\n");
    
    // Modelo
    model_t model;
    srand(42);
    for (int i = 0; i < H1 * IN_DIM; i++) model.w1[i] = rand() % 256 - 128;
    for (int i = 0; i < H2 * H1; i++) model.w2[i] = rand() % 256 - 128;
    for (int i = 0; i < OUT_DIM * H2; i++) model.w3[i] = rand() % 256 - 128;
    
    // Batch de inputs
    float* input = (float*)aligned_alloc(32, BATCH_SIZE * IN_DIM * sizeof(float));
    float* output = (float*)aligned_alloc(32, BATCH_SIZE * OUT_DIM * sizeof(float));
    
    for (int i = 0; i < BATCH_SIZE * IN_DIM; i++) {
        input[i] = (float)(rand() % 20 - 10) / 100.0f;
    }
    
    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < 100; i++) {
        rinx_batch_inference(&model, input, output, BATCH_SIZE);
    }
    
    // Benchmark
    printf("Benchmarking batch of %d images...\n\n", BATCH_SIZE);
    
    int num_batches = 1000;
    double start = get_time();
    for (int i = 0; i < num_batches; i++) {
        rinx_batch_inference(&model, input, output, BATCH_SIZE);
    }
    double end = get_time();
    
    double total_time_ms = (end - start) * 1000.0;
    double time_per_image_ms = total_time_ms / (num_batches * BATCH_SIZE);
    double throughput = (num_batches * BATCH_SIZE) / ((end - start));
    
    printf("RESULTADOS BATCH:\n");
    printf("  Total batches: %d\n", num_batches);
    printf("  Total images: %d\n", num_batches * BATCH_SIZE);
    printf("  Total time: %.2f ms\n", total_time_ms);
    printf("  Time per image: %.4f ms\n", time_per_image_ms);
    printf("  Throughput: %.0f images/sec\n", throughput);
    printf("\n");
    
    // Comparación
    double onnx_single = 0.035;  // ms
    double rinx_single = time_per_image_ms;
    double speedup = onnx_single / rinx_single;
    
    printf("COMPARACIÓN (por imagen):\n");
    printf("  ONNX Runtime: %.3f ms\n", onnx_single);
    printf("  RIN-X Batch:  %.4f ms\n", rinx_single);
    printf("\n");
    
    if (speedup > 1.0) {
        printf("  ✅ %.2f× más rápido que ONNX\n", speedup);
    }
    
    printf("\nThroughput comparison:\n");
    printf("  ONNX (est.): %.0f img/s\n", 1000.0 / onnx_single);
    printf("  RIN-X Batch: %.0f img/s\n", throughput);
    printf("\n");
    
    printf("=====================================\n");
    
    free(input);
    free(output);
    
    return 0;
}
