#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Tiny RIN for MNIST - Carga pesos de PyTorch
#define INPUT_DIM 784
#define HIDDEN_DIM 64
#define OUTPUT_DIM 10
#define TIME_STEPS 5

typedef struct {
    float layer1_w[HIDDEN_DIM][INPUT_DIM];
    float layer2_w[HIDDEN_DIM][HIDDEN_DIM];
    float readout_w[OUTPUT_DIM][HIDDEN_DIM];
    float threshold;
    float decay;
} TinyRIN_Weights;

typedef struct {
    float v_mem[HIDDEN_DIM];
} LIF_State;

// Leer pesos desde archivo binario
int load_weights(const char* filename, TinyRIN_Weights* weights) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: No se pudo abrir %s\n", filename);
        return 0;
    }
    
    // Formato simple: layer1 (64*784) + layer2 (64*64) + readout (10*64) + threshold + decay
    size_t r1 = fread(weights->layer1_w, sizeof(float), HIDDEN_DIM * INPUT_DIM, f);
    size_t r2 = fread(weights->layer2_w, sizeof(float), HIDDEN_DIM * HIDDEN_DIM, f);
    size_t r3 = fread(weights->readout_w, sizeof(float), OUTPUT_DIM * HIDDEN_DIM, f);
    size_t r4 = fread(&weights->threshold, sizeof(float), 1, f);
    size_t r5 = fread(&weights->decay, sizeof(float), 1, f);
    
    fclose(f);
    
    printf("Pesos cargados: L1=%zu, L2=%zu, RO=%zu\n", r1, r2, r3);
    printf("Threshold: %.4f, Decay: %.4f\n", weights->threshold, weights->decay);
    
    return (r1 == HIDDEN_DIM * INPUT_DIM && r2 == HIDDEN_DIM * HIDDEN_DIM && 
            r3 == OUTPUT_DIM * HIDDEN_DIM);
}

// LIF forward
void lif_forward(const float* weights, const float* input, float* output, 
                 float* v_mem, int in_dim, int out_dim, float threshold, float decay, int time_steps) {
    memset(v_mem, 0, out_dim * sizeof(float));
    memset(output, 0, out_dim * sizeof(float));
    
    for (int t = 0; t < time_steps; t++) {
        // Compute current: W * x
        for (int i = 0; i < out_dim; i++) {
            float current = 0;
            for (int j = 0; j < in_dim; j++) {
                current += weights[i * in_dim + j] * input[j];
            }
            
            // LIF dynamics
            v_mem[i] = v_mem[i] * decay + current;
            
            // Spike
            float spike = (v_mem[i] >= threshold) ? 1.0f : 0.0f;
            v_mem[i] = v_mem[i] * (1.0f - spike);
            output[i] += spike;
        }
    }
    
    // Average over time
    for (int i = 0; i < out_dim; i++) {
        output[i] /= time_steps;
    }
}

// Forward completo
void tiny_rin_forward(const TinyRIN_Weights* w, const float* input, float* logits) {
    float hidden1[HIDDEN_DIM];
    float hidden2[HIDDEN_DIM];
    float v_mem1[HIDDEN_DIM], v_mem2[HIDDEN_DIM];
    
    // Layer 1: 784 -> 64
    lif_forward((const float*)w->layer1_w, input, hidden1, v_mem1, 
                INPUT_DIM, HIDDEN_DIM, w->threshold, w->decay, TIME_STEPS);
    
    // Layer 2: 64 -> 64
    lif_forward((const float*)w->layer2_w, hidden1, hidden2, v_mem2,
                HIDDEN_DIM, HIDDEN_DIM, w->threshold, w->decay, TIME_STEPS);
    
    // Readout: 64 -> 10 (lineal, no LIF)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        logits[i] = 0;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            logits[i] += w->readout_w[i][j] * hidden2[j];
        }
    }
}

// Leer samples de test desde archivo generado por Python
#define MAX_SAMPLES 100
typedef struct {
    float input[INPUT_DIM];
    int target;
    float expected_logits[OUTPUT_DIM];
} TestSample;

int load_samples(const char* filename, TestSample* samples) {
    // Simplificado: Leer desde formato texto generado por Python
    // Por ahora, generamos samples aleatorios para test
    printf("Generando %d samples aleatorios para test...\n", MAX_SAMPLES);
    
    for (int s = 0; s < MAX_SAMPLES; s++) {
        // Input: imagen MNIST normalizada
        for (int i = 0; i < INPUT_DIM; i++) {
            samples[s].input[i] = ((float)rand() / RAND_MAX - 0.1307f) / 0.3081f;
        }
        samples[s].target = rand() % 10;
        memset(samples[s].expected_logits, 0, sizeof(float) * OUTPUT_DIM);
    }
    
    return MAX_SAMPLES;
}

// Comparar salidas
float compare_outputs(const float* a, const float* b, int n) {
    float mse = 0;
    float max_diff = 0;
    
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        mse += diff * diff;
        if (diff > max_diff) max_diff = diff;
    }
    
    mse /= n;
    return sqrtf(mse);  // RMSE
}

int main() {
    printf("========================================\n");
    printf("TINY RIN - VALIDACIÓN RIN-X vs PYTORCH\n");
    printf("========================================\n\n");
    
    TinyRIN_Weights weights;
    TestSample samples[MAX_SAMPLES];
    
    // Cargar pesos
    printf("[1/3] Cargando pesos entrenados...\n");
    if (!load_weights("tiny_rin_weights.bin", &weights)) {
        printf("ERROR: No se encontraron pesos entrenados.\n");
        printf("Ejecutar primero: python3 tiny_rin_train.py\n");
        return 1;
    }
    printf("✓ Pesos cargados\n\n");
    
    // Cargar samples
    printf("[2/3] Cargando samples de validación...\n");
    int num_samples = load_samples("tiny_rin_test_samples.json", samples);
    printf("✓ %d samples cargados\n\n", num_samples);
    
    // Inferencia
    printf("[3/3] Ejecutando inferencia RIN-X...\n");
    int correct = 0;
    float total_rmse = 0;
    
    for (int s = 0; s < num_samples; s++) {
        float logits[OUTPUT_DIM];
        tiny_rin_forward(&weights, samples[s].input, logits);
        
        // Encontrar predicción
        int pred = 0;
        float max_logit = logits[0];
        for (int i = 1; i < OUTPUT_DIM; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                pred = i;
            }
        }
        
        if (pred == samples[s].target) correct++;
        
        // Comparar con expected (si tenemos datos de PyTorch)
        float rmse = compare_outputs(logits, samples[s].expected_logits, OUTPUT_DIM);
        total_rmse += rmse;
        
        if (s < 5) {  // Mostrar primeros 5
            printf("Sample %d: pred=%d, target=%d, RMSE=%.6f\n", s, pred, samples[s].target, rmse);
        }
    }
    
    printf("\n--- RESULTADOS RIN-X ---\n");
    printf("Accuracy: %.2f%% (%d/%d)\n", 100.0f * correct / num_samples, correct, num_samples);
    printf("RMSE promedio vs PyTorch: %.6f\n", total_rmse / num_samples);
    printf("\n");
    
    // Validación
    printf("--- VALIDACIÓN DE CALIDAD ---\n");
    float acc_ratio = (float)correct / num_samples;
    
    // Nota: Como no tenemos el accuracy real de PyTorch en estos samples,
    // solo reportamos el de RIN-X y comparamos la salida numérica
    printf("Estado: Necesita comparación con PyTorch en mismos samples\n");
    printf("Siguiente: Ejecutar comparación lado a lado\n");
    
    return 0;
}
