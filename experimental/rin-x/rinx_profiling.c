/*
 * RIN-X PROFILING - Identificación de cuellos de botella
 * Instrumentación detallada de cada etapa del inference
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

#define THRESHOLD 0.5f
#define DECAY 0.8f
#define SCALE 0.01f

// Estructura para métricas de profiling
typedef struct {
    double gemv_w1_time;
    double lif_w1_time;
    double gemv_w2_time;
    double lif_w2_time;
    double gemv_w3_time;
    double total_time;
    double overhead_time;
} profile_metrics_t;

// Modelo
typedef struct {
    int8_t w1[H1 * IN_DIM];
    int8_t w2[H2 * H1];
    int8_t w3[OUT_DIM * H2];
} model_t;

// High-resolution timer
static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// GEMV con profiling
static void int8_gemv_profile(const int8_t* A, const float* x, float* y, 
                               int rows, int cols, double* time_out) {
    double start = get_time_ns();
    
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (int32_t)A[i * cols + j] * (int32_t)(x[j] * 100);
        }
        y[i] = (float)sum * SCALE / 100.0f;
    }
    
    *time_out = (get_time_ns() - start) / 1e6; // Convertir a ms
}

// LIF con profiling
static void lif_forward_profile(const int8_t* w, const float* x, float* out,
                                 int rows, int cols, float* v_mem, int ts,
                                 double* gemv_time, double* lif_time) {
    float* current = (float*)alloca(rows * sizeof(float));
    
    double gemv_total = 0;
    double lif_total = 0;
    
    memset(v_mem, 0, rows * sizeof(float));
    
    for (int t = 0; t < ts; t++) {
        // GEMV
        double gemv_start = get_time_ns();
        for (int i = 0; i < rows; i++) {
            int32_t sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += (int32_t)w[i * cols + j] * (int32_t)(x[j] * 100);
            }
            current[i] = (float)sum * SCALE / 100.0f;
        }
        gemv_total += (get_time_ns() - gemv_start) / 1e6;
        
        // LIF dynamics
        double lif_start = get_time_ns();
        for (int i = 0; i < rows; i++) {
            v_mem[i] = v_mem[i] * DECAY + current[i];
            if (v_mem[i] >= THRESHOLD) v_mem[i] = 0.0f;
        }
        lif_total += (get_time_ns() - lif_start) / 1e6;
    }
    
    *gemv_time = gemv_total;
    *lif_time = lif_total;
    
    for (int i = 0; i < rows; i++) {
        out[i] = v_mem[i] / ts;
    }
}

// Inference con profiling completo
void rinx_inference_profile(const model_t* m, const float* input, float* output,
                             profile_metrics_t* metrics) {
    float h1[H1], h2[H2];
    float v1[H1], v2[H2];
    
    double total_start = get_time_ns();
    
    // Capa 1
    double gemv_w1, lif_w1;
    lif_forward_profile(m->w1, input, h1, H1, IN_DIM, v1, TIME_STEPS, &gemv_w1, &lif_w1);
    metrics->gemv_w1_time = gemv_w1;
    metrics->lif_w1_time = lif_w1;
    
    // Capa 2
    double gemv_w2, lif_w2;
    lif_forward_profile(m->w2, h1, h2, H2, H1, v2, TIME_STEPS, &gemv_w2, &lif_w2);
    metrics->gemv_w2_time = gemv_w2;
    metrics->lif_w2_time = lif_w2;
    
    // Capa 3 (sin LIF, solo GEMV)
    double gemv_start = get_time_ns();
    for (int i = 0; i < OUT_DIM; i++) {
        int32_t sum = 0;
        for (int j = 0; j < H2; j++) {
            sum += (int32_t)m->w3[i * H2 + j] * (int32_t)(h2[j] * 100);
        }
        output[i] = (float)sum * SCALE / 100.0f;
    }
    metrics->gemv_w3_time = (get_time_ns() - gemv_start) / 1e6;
    
    metrics->total_time = (get_time_ns() - total_start) / 1e6;
    
    // Calcular overhead (loop, function calls, etc.)
    double computed_time = metrics->gemv_w1_time + metrics->lif_w1_time + 
                         metrics->gemv_w2_time + metrics->lif_w2_time + 
                         metrics->gemv_w3_time;
    metrics->overhead_time = metrics->total_time - computed_time;
}

// Inicialización
void init_model(model_t* m) {
    srand(42);
    for (int i = 0; i < H1 * IN_DIM; i++) m->w1[i] = rand() % 256 - 128;
    for (int i = 0; i < H2 * H1; i++) m->w2[i] = rand() % 256 - 128;
    for (int i = 0; i < OUT_DIM * H2; i++) m->w3[i] = rand() % 256 - 128;
}

// MAIN
int main() {
    printf("=====================================\n");
    printf("RIN-X PROFILING - Análisis de Performance\n");
    printf("=====================================\n\n");
    
    model_t model;
    init_model(&model);
    
    float input[IN_DIM];
    float output[OUT_DIM];
    for (int i = 0; i < IN_DIM; i++) input[i] = (rand() % 20 - 10) / 100.0f;
    
    profile_metrics_t metrics;
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        rinx_inference_profile(&model, input, output, &metrics);
    }
    
    // Benchmark con profiling
    int runs = 1000;
    profile_metrics_t total_metrics = {0};
    
    for (int i = 0; i < runs; i++) {
        rinx_inference_profile(&model, input, output, &metrics);
        total_metrics.gemv_w1_time += metrics.gemv_w1_time;
        total_metrics.lif_w1_time += metrics.lif_w1_time;
        total_metrics.gemv_w2_time += metrics.gemv_w2_time;
        total_metrics.lif_w2_time += metrics.lif_w2_time;
        total_metrics.gemv_w3_time += metrics.gemv_w3_time;
        total_metrics.total_time += metrics.total_time;
        total_metrics.overhead_time += metrics.overhead_time;
    }
    
    // Promedios
    double avg_total = total_metrics.total_time / runs;
    double avg_gemv_w1 = total_metrics.gemv_w1_time / runs;
    double avg_lif_w1 = total_metrics.lif_w1_time / runs;
    double avg_gemv_w2 = total_metrics.gemv_w2_time / runs;
    double avg_lif_w2 = total_metrics.lif_w2_time / runs;
    double avg_gemv_w3 = total_metrics.gemv_w3_time / runs;
    double avg_overhead = total_metrics.overhead_time / runs;
    
    printf("BREAKDOWN DE TIEMPOS (promedio de %d runs):\n", runs);
    printf("--------------------------------------------------\n");
    printf("\n");
    printf("Capa 1 (784→128):\n");
    printf("  GEMV:  %.4f ms (%.1f%%)\n", avg_gemv_w1, avg_gemv_w1/avg_total*100);
    printf("  LIF:   %.4f ms (%.1f%%)\n", avg_lif_w1, avg_lif_w1/avg_total*100);
    printf("  Total: %.4f ms\n", avg_gemv_w1 + avg_lif_w1);
    printf("\n");
    
    printf("Capa 2 (128→128):\n");
    printf("  GEMV:  %.4f ms (%.1f%%)\n", avg_gemv_w2, avg_gemv_w2/avg_total*100);
    printf("  LIF:   %.4f ms (%.1f%%)\n", avg_lif_w2, avg_lif_w2/avg_total*100);
    printf("  Total: %.4f ms\n", avg_gemv_w2 + avg_lif_w2);
    printf("\n");
    
    printf("Capa 3 (128→10):\n");
    printf("  GEMV:  %.4f ms (%.1f%%)\n", avg_gemv_w3, avg_gemv_w3/avg_total*100);
    printf("\n");
    
    printf("Overhead (loops, func calls):\n");
    printf("  %.4f ms (%.1f%%)\n", avg_overhead, avg_overhead/avg_total*100);
    printf("\n");
    
    printf("TOTAL: %.4f ms\n", avg_total);
    printf("--------------------------------------------------\n");
    printf("\n\n");
    
    // Análisis de cuellos de botella
    printf("ANÁLISIS DE CUELLOS DE BOTELLA:\n");
    printf("================================\n\n");
    
    double total_gemv = avg_gemv_w1 + avg_gemv_w2 + avg_gemv_w3;
    double total_lif = avg_lif_w1 + avg_lif_w2;
    
    printf("Tiempo total en GEMV:  %.4f ms (%.1f%%)\n", total_gemv, total_gemv/avg_total*100);
    printf("Tiempo total en LIF:   %.4f ms (%.1f%%)\n", total_lif, total_lif/avg_total*100);
    printf("Overhead:              %.4f ms (%.1f%%)\n", avg_overhead, avg_overhead/avg_total*100);
    printf("\n");
    
    // Identificar bottleneck principal
    if (total_gemv > total_lif && total_gemv > avg_overhead) {
        printf("🔍 BOTTLENECK PRINCIPAL: GEMV (%.1f%% del tiempo)\n", total_gemv/avg_total*100);
        printf("   Optimización recomendada:\n");
        printf("   - AVX2 intensivo con _mm256_madd_epi16\n");
        printf("   - Loop unrolling 4x o 8x\n");
        printf("   - Prefetching de pesos\n");
        printf("   - Bloques de 32 o 64 elementos\n");
    } else if (total_lif > avg_overhead) {
        printf("🔍 BOTTLENECK PRINCIPAL: LIF dynamics\n");
        printf("   Optimización recomendada:\n");
        printf("   - Vectorizar LIF con AVX2\n");
        printf("   - Fusionar GEMV+LIF en un loop\n");
    } else {
        printf("🔍 BOTTLENECK PRINCIPAL: Overhead de funciones/loops\n");
        printf("   Optimización recomendada:\n");
        printf("   - Inline de funciones\n");
        printf("   - Reducir llamadas a funciones\n");
        printf("   - Unroll de loops externos\n");
    }
    
    printf("\n");
    
    // Proyección de optimización
    printf("PROYECCIÓN DE OPTIMIZACIÓN:\n");
    printf("==========================\n\n");
    
    double gemv_optimized = total_gemv * 0.6;  // 40% mejora con AVX2 óptimo
    double lif_optimized = total_lif * 0.7;     // 30% mejora
    double overhead_optimized = avg_overhead * 0.5;  // 50% mejora con inlining
    
    double projected_total = gemv_optimized + lif_optimized + overhead_optimized;
    double speedup = avg_total / projected_total;
    
    printf("Tiempo actual:        %.4f ms\n", avg_total);
    printf("Tiempo proyectado:    %.4f ms\n", projected_total);
    printf("Speedup esperado:     %.2f×\n", speedup);
    printf("\n");
    
    // Comparación con ONNX
    double onnx_time = 0.035;
    double current_speedup = onnx_time / avg_total;
    double projected_speedup = onnx_time / projected_total;
    
    printf("Comparación vs ONNX (%.3f ms):\n", onnx_time);
    printf("  Actual:    %.2f×\n", current_speedup);
    printf("  Proyectado: %.2f× (objetivo: 3.0-4.0×)\n", projected_speedup);
    printf("\n");
    
    if (projected_speedup >= 3.0) {
        printf("✅ Con optimizaciones proyectadas, se alcanza objetivo de 3×+\n");
    } else {
        printf("⚠️  Necesitarías más optimizaciones para llegar a 3×\n");
        printf("   Considera:\n");
        printf("   - Quantization de activaciones también (no solo pesos)\n");
        printf("   - Kernel fusion más agresivo\n");
        printf("   - Hand-optimized assembly\n");
    }
    
    printf("\n=====================================\n");
    printf("Profiling completado.\n");
    printf("=====================================\n");
    
    return 0;
}
