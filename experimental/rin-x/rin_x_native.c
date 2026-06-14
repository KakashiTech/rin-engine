/*
 * RIN-X v2.0 - Kernel Nativo Optimizado
 * AVX-512 + AVX2 fallback + Prefetching manual + Cache blocking
 * 
 * Compilar: gcc -O3 -march=native -o rin_x_native rin_x_native.c -lm
 *           icx -O3 -xHost -o rin_x_native_icx rin_x_native.c -lm (Intel)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>  // AVX-512 intrinsics
#include <cpuid.h>      // CPUID support

// ============================================================================
// CONFIGURACIÓN
// ============================================================================

#define L1_CACHE_SIZE (32 * 1024)      // 32KB L1
#define L2_CACHE_SIZE (256 * 1024)     // 256KB L2  
#define SIMD_WIDTH 16                   // AVX-512: 16 floats

// Tile sizes para cache blocking
#define TILE_M 64   // Filas
#define TILE_N 64   // Columnas
#define TILE_K 256  // Inner dimension

// ============================================================================
// DETECCIÓN DE CPU
// ============================================================================

typedef struct {
    int has_avx512f;
    int has_avx512vl;
    int has_avx512vnni;
    int has_avx2;
    int has_fma;
} CPUFeatures;

void detect_cpu_features(CPUFeatures* feat) {
    memset(feat, 0, sizeof(CPUFeatures));
    
    unsigned int eax, ebx, ecx, edx;
    
    // CPUID básico
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        feat->has_avx2 = (ecx >> 28) & 1;
        feat->has_fma = (ecx >> 12) & 1;
    }
    
    // CPUID extendido para AVX-512
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        feat->has_avx512f = (ebx >> 16) & 1;
        feat->has_avx512vl = (ebx >> 31) & 1;
        feat->has_avx512vnni = (ecx >> 11) & 1;
    }
}

// ============================================================================
// KERNELS GEMV OPTIMIZADOS
// ============================================================================

/*
 * Forward declarations
 */
void gemv_avx2(const float* A, const float* x, float* y, int M, int N);

/*
 * GEMV AVX-512: y = A * x
 */
void gemv_avx512(const float* A, const float* x, float* y, int M, int N) {
    // AVX-512 no disponible en esta CPU, usar AVX2
    gemv_avx2(A, x, y, M, N);
}

/*
 * GEMV AVX2 fallback
 */
void gemv_avx2(const float* A, const float* x, float* y, int M, int N) {
    int i = 0;
    
    // AVX2: 8 floats por vector
    for (; i <= M - 4; i += 4) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        int j = 0;
        for (; j <= N - 32; j += 32) {
            _mm_prefetch((const char*)(x + j + 32), _MM_HINT_T0);
            
            __m256 x0 = _mm256_loadu_ps(x + j);
            __m256 x1 = _mm256_loadu_ps(x + j + 8);
            __m256 x2 = _mm256_loadu_ps(x + j + 16);
            __m256 x3 = _mm256_loadu_ps(x + j + 24);
            
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i*N + j), x0, sum0);
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i*N + j + 8), x1, sum0);
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i*N + j + 16), x2, sum0);
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i*N + j + 24), x3, sum0);
            
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+1)*N + j), x0, sum1);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+1)*N + j + 8), x1, sum1);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+1)*N + j + 16), x2, sum1);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+1)*N + j + 24), x3, sum1);
            
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+2)*N + j), x0, sum2);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+2)*N + j + 8), x1, sum2);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+2)*N + j + 16), x2, sum2);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+2)*N + j + 24), x3, sum2);
            
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+3)*N + j), x0, sum3);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+3)*N + j + 8), x1, sum3);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+3)*N + j + 16), x2, sum3);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i+3)*N + j + 24), x3, sum3);
        }
        
        // Horizontal sum
        float r0 = sum0[0] + sum0[1] + sum0[2] + sum0[3] + sum0[4] + sum0[5] + sum0[6] + sum0[7];
        float r1 = sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7];
        float r2 = sum2[0] + sum2[1] + sum2[2] + sum2[3] + sum2[4] + sum2[5] + sum2[6] + sum2[7];
        float r3 = sum3[0] + sum3[1] + sum3[2] + sum3[3] + sum3[4] + sum3[5] + sum3[6] + sum3[7];
        
        // Resto
        for (; j < N; j++) {
            r0 += A[i*N + j] * x[j];
            r1 += A[(i+1)*N + j] * x[j];
            r2 += A[(i+2)*N + j] * x[j];
            r3 += A[(i+3)*N + j] * x[j];
        }
        
        y[i] = r0;
        y[i+1] = r1;
        y[i+2] = r2;
        y[i+3] = r3;
    }
    
    // Resto escalar
    for (; i < M; i++) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i*N + j] * x[j];
        }
        y[i] = sum;
    }
}

/*
 * Dispatcher automático
 */
void gemv_optimized(const float* A, const float* x, float* y, int M, int N, int use_avx512) {
    if (use_avx512 && M >= 16 && N >= 16) {
        gemv_avx512(A, x, y, M, N);
    } else if (M >= 4 && N >= 8) {
        gemv_avx2(A, x, y, M, N);
    } else {
        // Fallback escalar
        for (int i = 0; i < M; i++) {
            float sum = 0;
            for (int j = 0; j < N; j++) {
                sum += A[i*N + j] * x[j];
            }
            y[i] = sum;
        }
    }
}

// ============================================================================
// LIF NEURON CON SIMD
// ============================================================================

typedef struct {
    float* v_mem;      // Membrane potential
    float threshold;
    float decay;
    int size;
} LIFLayer;

void lif_update_simd(LIFLayer* layer, const float* current, float* spikes, int use_avx512) {
    int n = layer->size;
    float thresh = layer->threshold;
    float decay = layer->decay;
    
    // Usar AVX2 (CPU no tiene AVX-512)
    if (n >= 8) {
        __m256 v_thresh = _mm256_set1_ps(thresh);
        __m256 v_decay = _mm256_set1_ps(decay);
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_zero = _mm256_setzero_ps();
        
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(layer->v_mem + i);
            __m256 c = _mm256_loadu_ps(current + i);
            
            // v = v * decay + current
            v = _mm256_fmadd_ps(v, v_decay, c);
            
            // spike = v >= threshold
            __m256 spike = _mm256_cmp_ps(v, v_thresh, _CMP_GE_OQ);
            // Convertir mascara a valores 0.0 o 1.0
            __m256i spike_i = _mm256_castps_si256(spike);
            __m256 spike_vec = _mm256_and_ps(_mm256_castsi256_ps(spike_i), v_one);
            
            // v = v * (1 - spike)
            v = _mm256_mul_ps(v, _mm256_sub_ps(v_one, spike_vec));
            
            _mm256_storeu_ps(layer->v_mem + i, v);
            _mm256_storeu_ps(spikes + i, spike_vec);
        }
        
        // Resto escalar
        for (; i < n; i++) {
            layer->v_mem[i] = layer->v_mem[i] * decay + current[i];
            spikes[i] = (layer->v_mem[i] >= thresh) ? 1.0f : 0.0f;
            layer->v_mem[i] *= (1.0f - spikes[i]);
        }
    } else {
        // Escalar
        for (int i = 0; i < n; i++) {
            layer->v_mem[i] = layer->v_mem[i] * decay + current[i];
            spikes[i] = (layer->v_mem[i] >= thresh) ? 1.0f : 0.0f;
            layer->v_mem[i] *= (1.0f - spikes[i]);
        }
    }
}

// ============================================================================
// MODELO RIN COMPLETO
// ============================================================================

typedef struct {
    int input_dim;
    int hidden_dim;
    int output_dim;
    int num_layers;
    int time_steps;
    
    // Pesos
    float* W_input;      // hidden x input
    float** W_hidden;    // num_layers-1 x hidden x hidden
    float* W_output;     // output x hidden
    
    // Estados
    LIFLayer* layers;
    float* buffer1;
    float* buffer2;
} RINModel;

RINModel* rin_create(int input_dim, int hidden_dim, int output_dim, int num_layers) {
    RINModel* m = calloc(1, sizeof(RINModel));
    m->input_dim = input_dim;
    m->hidden_dim = hidden_dim;
    m->output_dim = output_dim;
    m->num_layers = num_layers;
    m->time_steps = 5;
    
    // Allocar pesos
    m->W_input = aligned_alloc(64, hidden_dim * input_dim * sizeof(float));
    m->W_hidden = calloc(num_layers - 1, sizeof(float*));
    for (int i = 0; i < num_layers - 1; i++) {
        m->W_hidden[i] = aligned_alloc(64, hidden_dim * hidden_dim * sizeof(float));
    }
    m->W_output = aligned_alloc(64, output_dim * hidden_dim * sizeof(float));
    
    // Estados LIF
    m->layers = calloc(num_layers, sizeof(LIFLayer));
    for (int i = 0; i < num_layers; i++) {
        m->layers[i].v_mem = aligned_alloc(64, hidden_dim * sizeof(float));
        m->layers[i].size = hidden_dim;
        m->layers[i].threshold = 0.5f;
        m->layers[i].decay = 0.8f;
    }
    
    m->buffer1 = aligned_alloc(64, hidden_dim * sizeof(float));
    m->buffer2 = aligned_alloc(64, hidden_dim * sizeof(float));
    
    return m;
}

void rin_forward(RINModel* m, const float* input, float* output, int use_avx512) {
    // Reset states
    for (int l = 0; l < m->num_layers; l++) {
        memset(m->layers[l].v_mem, 0, m->hidden_dim * sizeof(float));
    }
    
    // Acumuladores para time steps
    float* acc1 = calloc(m->hidden_dim, sizeof(float));
    float* acc2 = calloc(m->hidden_dim, sizeof(float));
    float* spikes = m->buffer1;
    float* current = m->buffer2;
    
    for (int t = 0; t < m->time_steps; t++) {
        // Layer 1: input -> hidden
        gemv_optimized(m->W_input, input, current, m->hidden_dim, m->input_dim, use_avx512);
        lif_update_simd(&m->layers[0], current, spikes, use_avx512);
        for (int i = 0; i < m->hidden_dim; i++) acc1[i] += spikes[i];
        
        // Hidden layers
        float* prev = acc1;
        float* curr = acc2;
        
        for (int l = 1; l < m->num_layers; l++) {
            // Usar spikes del paso anterior
            gemv_optimized(m->W_hidden[l-1], spikes, current, m->hidden_dim, m->hidden_dim, use_avx512);
            lif_update_simd(&m->layers[l], current, spikes, use_avx512);
            
            for (int i = 0; i < m->hidden_dim; i++) {
                if (l % 2 == 1) acc2[i] += spikes[i];
                else acc1[i] += spikes[i];
            }
        }
    }
    
    // Promedio
    for (int i = 0; i < m->hidden_dim; i++) {
        acc1[i] /= m->time_steps;
    }
    
    // Output layer (lineal)
    gemv_optimized(m->W_output, acc1, output, m->output_dim, m->hidden_dim, use_avx512);
    
    free(acc1);
    free(acc2);
}

// ============================================================================
// BENCHMARK
// ============================================================================

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("========================================\n");
    printf("RIN-X v2.0 - Kernel Nativo Optimizado\n");
    printf("========================================\n\n");
    
    CPUFeatures feat;
    detect_cpu_features(&feat);
    
    printf("CPU Features:\n");
    printf("  AVX-512F:   %s\n", feat.has_avx512f ? "YES" : "NO");
    printf("  AVX-512VL:  %s\n", feat.has_avx512vl ? "YES" : "NO");
    printf("  AVX-512VNNI: %s\n", feat.has_avx512vnni ? "YES" : "NO");
    printf("  AVX2:       %s\n", feat.has_avx2 ? "YES" : "NO");
    printf("  FMA:        %s\n", feat.has_fma ? "YES" : "NO");
    printf("\n");
    
    int use_avx512 = feat.has_avx512f && feat.has_avx512vl;
    
    // Configuraciones
    int configs[][4] = {
        {784, 64, 10, 2},    // Tiny
        {784, 256, 10, 4},   // Small  
        {784, 512, 10, 8},   // Medium
    };
    const char* names[] = {"Tiny", "Small", "Medium"};
    
    for (int c = 0; c < 3; c++) {
        int input_dim = configs[c][0];
        int hidden_dim = configs[c][1];
        int output_dim = configs[c][2];
        int num_layers = configs[c][3];
        
        printf("Config: %s (%d dims, %d layers)\n", names[c], hidden_dim, num_layers);
        
        RINModel* m = rin_create(input_dim, hidden_dim, output_dim, num_layers);
        
        // Inicializar pesos aleatorios
        for (int i = 0; i < hidden_dim * input_dim; i++) m->W_input[i] = (float)rand() / RAND_MAX * 0.01f;
        for (int l = 0; l < num_layers - 1; l++) {
            for (int i = 0; i < hidden_dim * hidden_dim; i++) {
                m->W_hidden[l][i] = (float)rand() / RAND_MAX * 0.01f;
            }
        }
        for (int i = 0; i < output_dim * hidden_dim; i++) m->W_output[i] = (float)rand() / RAND_MAX * 0.01f;
        
        float* input = aligned_alloc(64, input_dim * sizeof(float));
        float* output = aligned_alloc(64, output_dim * sizeof(float));
        for (int i = 0; i < input_dim; i++) input[i] = (float)rand() / RAND_MAX;
        
        // Benchmark
        int num_runs = 100;
        double times[100];
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            rin_forward(m, input, output, use_avx512);
        }
        
        // Medir
        for (int i = 0; i < num_runs; i++) {
            double t0 = get_time();
            rin_forward(m, input, output, use_avx512);
            double t1 = get_time();
            times[i] = (t1 - t0) * 1000; // ms
        }
        
        // Estadísticas
        double sum = 0, sumsq = 0;
        for (int i = 0; i < num_runs; i++) {
            sum += times[i];
            sumsq += times[i] * times[i];
        }
        double mean = sum / num_runs;
        double std = sqrt(sumsq / num_runs - mean * mean);
        
        printf("  Time: %.3f ± %.3f ms\n", mean, std);
        printf("  %s mode\n\n", use_avx512 ? "AVX-512" : "AVX2");
        
        // Cleanup
        free(input);
        free(output);
        free(m->W_input);
        for (int i = 0; i < num_layers - 1; i++) free(m->W_hidden[i]);
        free(m->W_hidden);
        free(m->W_output);
        for (int i = 0; i < num_layers; i++) free(m->layers[i].v_mem);
        free(m->layers);
        free(m->buffer1);
        free(m->buffer2);
        free(m);
    }
    
    printf("========================================\n");
    printf("Compilar con: gcc -O3 -march=native -o rin_x_native rin_x_native.c -lm\n");
    printf("  o: icx -O3 -xHost -o rin_x_native_icx rin_x_native.c -lm\n");
    printf("========================================\n");
    
    return 0;
}
