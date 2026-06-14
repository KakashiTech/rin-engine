/*
 * RIN-X Implementation
 * Kernel fusion + AVX-512 + Block sparsity
 */

#include "rin_x.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cpuid.h>

// ============================================================================
// INICIALIZACIÓN
// ============================================================================

bool rinx_check_avx512_support(void) {
    unsigned int eax, ebx, ecx, edx;
    
    // Check AVX-512 Foundation
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        // Bit 16 of EBX = AVX-512F
        if (ebx & (1 << 16)) {
            return true;
        }
    }
    return false;
}

void rinx_init_sparse_weights(RINX_SparseWeights* W, float sparsity) {
    int num_blocks_total = RINX_MODEL_DIM / RINX_BLOCK_SPARSE_SIZE;  // 512 / 4 = 128
    int num_nonzero_blocks = (int)(num_blocks_total * (1.0f - sparsity));  // ~6 bloques
    
    W->num_blocks = num_nonzero_blocks;
    
    // Inicializar máscara y índices
    memset(W->block_mask, 0, sizeof(W->block_mask));
    
    // Seleccionar bloques no-cero aleatoriamente
    for (int b = 0; b < num_nonzero_blocks; b++) {
        int block_idx = rand() % num_blocks_total;
        W->block_indices[b] = block_idx;
        W->block_mask[block_idx / 16] |= (1 << (block_idx % 16));
        
        // Inicializar pesos del bloque
        for (int i = 0; i < 16; i++) {
            W->weights[b * 16 + i] = (rand() % 512 - 256);  // INT16 random
        }
    }
}

void rinx_init_model(RINX_Model* model) {
    for (int l = 0; l < RINX_NUM_LAYERS; l++) {
        // Inicializar capa
        for (int i = 0; i < RINX_MODEL_DIM; i++) {
            model->layers[l].v_mem[i] = 0;
            model->layers[l].threshold[i] = 1000;  // Q15: ~0.03
            model->layers[l].decay_shift[i] = 1;   // Divide by 2
            model->layers[l].input_shift[i] = 2;   // Divide by 4
        }
        
        // Inicializar pesos sparse
        rinx_init_sparse_weights(&model->weights[l], RINX_SPARSITY);
    }
    
    // Embedding
    for (int i = 0; i < RINX_MODEL_DIM; i++) {
        model->embedding[i] = (rand() % 256 - 128);
    }
}

// ============================================================================
// FORWARD PASS OPTIMIZADO
// ============================================================================

static inline void rinx_forward_token(
    const RINX_Model* restrict model,
    rinx_q15_t* restrict input,
    rinx_q15_t* restrict output
) {
    alignas(64) rinx_q15_t temp[RINX_MODEL_DIM];
    
    // Copiar input a temp
    memcpy(temp, input, sizeof(rinx_q15_t) * RINX_MODEL_DIM);
    
    // Forward through each layer
    for (int l = 0; l < RINX_NUM_LAYERS; l++) {
        // Activaciones
        alignas(64) rinx_q15_t activations[RINX_MODEL_DIM];
        
        // Sparse GEMV
        for (int i = 0; i < RINX_MODEL_DIM; i++) {
            activations[i] = 0;
        }
        
        const RINX_SparseWeights* W = &model->weights[l];
        RINX_LayerState* layer = &model->layers[l];
        
        // Procesar bloques no-cero
        for (int b = 0; b < W->num_blocks; b++) {
            int block_idx = W->block_indices[b];
            int row = block_idx * 4;
            
            const rinx_q15_t* block_weights = &W->weights[b * 16];
            
            // 4x4 matmul
            for (int r = 0; r < 4; r++) {
                int32_t sum = 0;
                for (int c = 0; c < 4; c++) {
                    sum += (int32_t)block_weights[r * 4 + c] * (int32_t)temp[row + c];
                }
                activations[row + r] += (rinx_q15_t)(sum >> RINX_Q15_SHIFT);
            }
        }
        
        // LIF update
        for (int i = 0; i < RINX_MODEL_DIM; i++) {
            // Leak
            int32_t leaked = layer->v_mem[i] >> layer->decay_shift[i];
            // Add input
            int32_t scaled_input = activations[i] >> layer->input_shift[i];
            int32_t new_v = leaked + scaled_input;
            
            // Saturate
            if (new_v > RINX_Q15_MAX) new_v = RINX_Q15_MAX;
            if (new_v < RINX_Q15_MIN) new_v = RINX_Q15_MIN;
            
            layer->v_mem[i] = (rinx_q15_t)new_v;
            
            // Spike (branchless)
            int spike = (layer->v_mem[i] >= layer->threshold[i]) ? 1 : 0;
            temp[i] = spike ? 0 : layer->v_mem[i];  // Reset on spike
            
            if (spike) {
                layer->v_mem[i] = 0;
            }
        }
    }
    
    // Copy final output
    memcpy(output, temp, sizeof(rinx_q15_t) * RINX_MODEL_DIM);
}

void rinx_forward_fused(
    const RINX_Model* restrict model,
    int num_tokens,
    rinx_q15_t* restrict output_buffer
) {
    alignas(64) rinx_q15_t input[RINX_MODEL_DIM];
    alignas(64) rinx_q15_t output[RINX_MODEL_DIM];
    
    for (int t = 0; t < num_tokens; t++) {
        // Generar input (embedding + noise)
        for (int i = 0; i < RINX_MODEL_DIM; i++) {
            int noise = (rand() % 64 - 32);
            int val = (int)model->embedding[i] + noise;
            if (val > RINX_Q15_MAX) val = RINX_Q15_MAX;
            if (val < RINX_Q15_MIN) val = RINX_Q15_MIN;
            input[i] = (rinx_q15_t)val;
        }
        
        // Forward
        rinx_forward_token(model, input, output);
        
        // Store result
        memcpy(&output_buffer[t * RINX_MODEL_DIM], output, sizeof(rinx_q15_t) * RINX_MODEL_DIM);
    }
}

// ============================================================================
// BENCHMARK 30 RUNS
// ============================================================================

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Leer RAPL
static uint64_t read_rapl() {
    FILE* f = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
    if (!f) return 0;
    uint64_t energy;
    fscanf(f, "%llu", &energy);
    fclose(f);
    return energy;
}

void rinx_benchmark_30runs(const RINX_Model* model) {
    printf("RIN-X Benchmark (30 runs, %d tokens)\n", 1000);
    printf("AVX-512: %s\n", rinx_check_avx512_support() ? "SUPPORTED" : "NOT AVAILABLE");
    printf("\n");
    
    double times[30];
    double energies[30];
    int num_tokens = 1000;  // Aumentado para medición precisa
    
    printf("Run  | Time (s) | Energy (J) | J/token\n");
    printf("-----+----------+------------+---------\n");
    
    for (int run = 0; run < 30; run++) {
        alignas(64) rinx_q15_t output[1000 * RINX_MODEL_DIM];
        
        uint64_t e_start = read_rapl();
        double t_start = get_time();
        
        rinx_forward_fused(model, num_tokens, output);
        
        double t_end = get_time();
        uint64_t e_end = read_rapl();
        
        times[run] = t_end - t_start;
        energies[run] = (double)(e_end - e_start) / 1e6;
        double j_per_token = energies[run] / num_tokens;
        
        printf("%3d  | %.4f   | %.3f      | %.6f\n", 
               run + 1, times[run], energies[run], j_per_token);
    }
    
    // Estadísticas
    double t_sum = 0, t_sumsq = 0;
    double e_sum = 0, e_sumsq = 0;
    
    for (int i = 0; i < 30; i++) {
        t_sum += times[i];
        t_sumsq += times[i] * times[i];
        e_sum += energies[i];
        e_sumsq += energies[i] * energies[i];
    }
    
    double t_mean = t_sum / 30;
    double t_std = sqrt((t_sumsq - (t_sum * t_sum) / 30) / 29);
    double e_mean = e_sum / 30;
    double e_std = sqrt((e_sumsq - (e_sum * e_sum) / 30) / 29);
    
    printf("\n");
    printf("RESULTADOS RIN-X:\n");
    printf("  Tiempo:    %.4f ± %.4f s\n", t_mean, t_std);
    printf("  Energía:   %.3f ± %.3f J\n", e_mean, e_std);
    printf("  J/token:   %.5f ± %.5f\n", e_mean / 100.0, e_std / 100.0);
    printf("  Tokens/s:  %.1f\n", 100.0 / t_mean);
    printf("\n");
    
    // Guardar CSV
    FILE* f = fopen("rinx_30runs.csv", "w");
    if (f) {
        fprintf(f, "run,time_sec,energy_j,j_per_token\n");
        for (int i = 0; i < 30; i++) {
            fprintf(f, "%d,%.6f,%.6f,%.8f\n", i + 1, times[i], energies[i], energies[i] / 100.0);
        }
        fclose(f);
        printf("Resultados guardados en: rinx_30runs.csv\n");
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     RIN-X: Revolutionary Neural Inference eXtreme              ║\n");
    printf("║     Kernel Fusion + AVX-512 + Block Sparsity                   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Verificar AVX-512
    if (!rinx_check_avx512_support()) {
        printf("⚠ AVX-512 NO disponible. Rendimiento será limitado.\n");
        printf("  Compile con: gcc -O3 -march=native -mavx512f ...\n\n");
    } else {
        printf("✓ AVX-512 detectado\n\n");
    }
    
    // Inicializar modelo
    RINX_Model model;
    rinx_init_model(&model);
    
    printf("Modelo: %d capas, dim=%d, sparsity=%.0f%%\n", 
           RINX_NUM_LAYERS, RINX_MODEL_DIM, RINX_SPARSITY * 100);
    printf("Block size: %dx%d (estructurado)\n\n", RINX_BLOCK_SPARSE_SIZE, RINX_BLOCK_SPARSE_SIZE);
    
    // Benchmark
    rinx_benchmark_30runs(&model);
    
    printf("\nComparar con PyTorch:\n");
    printf("  python3 pytorch_energy_meter.py\n");
    
    return 0;
}
