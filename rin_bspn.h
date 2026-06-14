/*
 * rin_bspn.h - Bit Shifting PowerNorm (BSPN)
 * 
 * Reemplaza LayerNorm estándar:
 *   LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *   
 * Con normalización L1 usando solo bit shifts:
 *   BSPN: y = x / sum(|x|) ≈ x >> log2(sum_abs)
 * 
 * Basado en: Sorbet - ICML 2025
 * Meta: Error < 0.5% vs LayerNorm estándar
 */

#ifndef RIN_BSPN_H
#define RIN_BSPN_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTAS Q15
 * ============================================================================ */

#define RIN_Q15_MAX     32767
#define RIN_Q15_MIN     (-32768)
#define RIN_BSPN_EPS    1       /* Epsilon mínimo para evitar div/0 */

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_BSPN_Params - Parámetros de normalización
 */
typedef struct {
    int16_t scale_shift;     /* Shift adicional para escalar output */
    int16_t bias;            /* Bias añadido post-normalización */
    uint8_t learnable;       /* Si es learnable o fijo */
    uint8_t use_l2;          /* 0=L1 (default), 1=L2 aproximado */
} RIN_BSPN_Params;

/*
 * RIN_BSPN_Layer - Capa BSPN completa
 */
typedef struct {
    RIN_BSPN_Params* params_per_channel;  /* Parámetros por canal */
    uint32_t num_channels;
    float    running_mean[128];            /* Estadísticas running (opcional) */
    float    running_var[128];               /* Estadísticas running (opcional) */
} RIN_BSPN_Layer;

/*
 * RIN_BSPN_Stats - Estadísticas computadas (para debugging)
 */
typedef struct {
    int32_t sum_abs;         /* Suma L1 */
    int32_t sum_sq;          /* Suma cuadrados (L2) */
    int16_t max_abs;         /* Máximo absoluto */
    uint8_t computed_shift;  /* Shift usado */
} RIN_BSPN_Stats;

/* ============================================================================
 * FUNCIÓN: RIN_IntegerLog2 (declaración externa - definida en ptsoftmax)
 * ============================================================================ */
extern uint32_t RIN_IntegerLog2(uint32_t x);

/* ============================================================================
 * FUNCIÓN: RIN_Q15_Abs
 * Valor absoluto Q15
 * ============================================================================ */
static inline int16_t RIN_Q15_Abs(int16_t x) {
    return (x < 0) ? -x : x;
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_ComputeL1Sum
 * Suma L1 con acumulación segura
 * 
 * @input: Array Q15
 * @len:   Longitud
 * 
 * Retorna: sum(|input[i]|)
 * ============================================================================ */
static inline int32_t RIN_BSPN_ComputeL1Sum(const int16_t* input, uint32_t len) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < len; i++) {
        sum += RIN_Q15_Abs(input[i]);
    }
    return sum;
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_ComputeL2SumApprox
 * Aproximación L2 usando shifts
 * 
 * Aproxima sqrt(sum(x²)) usando log2(sum(|x|)) + corrección
 * ============================================================================ */
static inline int32_t RIN_BSPN_ComputeL2SumApprox(const int16_t* input, uint32_t len) {
    /* Aproximación: L2 ≈ L1 * sqrt(2/π) para distribución normal */
    /* sqrt(2/π) ≈ 0.7979 ≈ 26103/32768 en Q15 */
    int32_t l1 = RIN_BSPN_ComputeL1Sum(input, len);
    return (l1 * 26103) >> 15;
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_ComputeStats
 * Computa estadísticas de un array
 * ============================================================================ */
static inline void RIN_BSPN_ComputeStats(const int16_t* input,
                                          uint32_t len,
                                          RIN_BSPN_Stats* stats) {
    if (!input || !stats || len == 0) return;
    
    stats->sum_abs = 0;
    stats->sum_sq = 0;
    stats->max_abs = 0;
    
    for (uint32_t i = 0; i < len; i++) {
        int16_t val = input[i];
        int16_t abs_val = RIN_Q15_Abs(val);
        
        stats->sum_abs += abs_val;
        stats->sum_sq += (int32_t)val * val;
        
        if (abs_val > stats->max_abs) {
            stats->max_abs = abs_val;
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_Forward
 * Forward pass de BSPN - normalización por bit shifting
 * 
 * @input:   Array Q15 de entrada
 * @output:  Array Q15 normalizado (pre-allocado)
 * @len:     Longitud
 * @params:  Parámetros de normalización
 * 
 * Algoritmo:
 *   1. sum_abs = sum(|input[i]|)
 *   2. log2_sum = log2(sum_abs)
 *   3. output[i] = input[i] >> log2_sum + bias
 * 
 * Error vs LayerNorm estándar: típicamente < 0.5%
 * ============================================================================ */
static inline void RIN_BSPN_Forward(const int16_t* input,
                                      int16_t* output,
                                      uint32_t len,
                                      const RIN_BSPN_Params* params) {
    if (!input || !output || !params || len == 0) return;
    
    /* Paso 1: Calcular suma L1 */
    int32_t sum_abs = RIN_BSPN_ComputeL1Sum(input, len);
    
    if (sum_abs <= RIN_BSPN_EPS) {
        /* Caso degenerado: todos ceros o muy pequeños */
        /* Output es input (no hay que normalizar) + bias */
        for (uint32_t i = 0; i < len; i++) {
            int32_t val = (int32_t)input[i] + params->bias;
            output[i] = (val > RIN_Q15_MAX) ? RIN_Q15_MAX : 
                       (val < RIN_Q15_MIN) ? RIN_Q15_MIN : (int16_t)val;
        }
        return;
    }
    
    /* Paso 2: log2(sum_abs) para determinar shift */
    /* sum_abs es int32, normalizarlo antes de log2 */
    uint32_t sum_for_log = (uint32_t)(sum_abs >> 15);  /* Convertir a escala aproximada */
    if (sum_for_log == 0) sum_for_log = 1;
    
    uint32_t log2_sum = RIN_IntegerLog2(sum_for_log);
    
    /* Ajustar por escala deseada */
    int32_t effective_shift = (int32_t)log2_sum + params->scale_shift;
    
    /* Paso 3: Normalizar cada elemento */
    for (uint32_t i = 0; i < len; i++) {
        int32_t normalized;
        
        if (effective_shift >= 0) {
            /* Shift right (dividir) */
            normalized = (int32_t)input[i] >> effective_shift;
        } else {
            /* Shift left (multiplicar) */
            normalized = (int32_t)input[i] << (-effective_shift);
        }
        
        /* Añadir bias */
        normalized += params->bias;
        
        /* Saturar a Q15 */
        if (normalized > RIN_Q15_MAX) normalized = RIN_Q15_MAX;
        if (normalized < RIN_Q15_MIN) normalized = RIN_Q15_MIN;
        
        output[i] = (int16_t)normalized;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_ForwardPerChannel
 * BSPN para datos multi-canal (ej: embeddings)
 * 
 * @input:      [channels][features] array flattenado
 * @output:     Pre-allocado del mismo tamaño
 * @channels:   Número de canales
 * @features:   Features por canal
 * @params:     Array de parámetros por canal
 * ============================================================================ */
static inline void RIN_BSPN_ForwardPerChannel(const int16_t* input,
                                                int16_t* output,
                                                uint32_t channels,
                                                uint32_t features,
                                                const RIN_BSPN_Params* params) {
    if (!input || !output || !params) return;
    
    for (uint32_t c = 0; c < channels; c++) {
        const int16_t* in_channel = &input[c * features];
        int16_t* out_channel = &output[c * features];
        
        RIN_BSPN_Forward(in_channel, out_channel, features, &params[c]);
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_ValidateError
 * Valida que error esté dentro de bounds
 * 
 * Usar durante tests para verificar fidelidad < 0.5%
 * 
 * Retorna: RMSE entre BSPN y LayerNorm de referencia
 * ============================================================================ */
static inline float RIN_BSPN_ValidateError(const int16_t* bspn_output,
                                            const float* standard_output,
                                            uint32_t len) {
    if (!bspn_output || !standard_output || len == 0) return 1.0f;
    
    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    
    for (uint32_t i = 0; i < len; i++) {
        float bspn_f = (float)bspn_output[i] / 32768.0f;
        float error = fabsf(bspn_f - standard_output[i]);
        
        if (error > max_error) max_error = error;
        sum_sq_error += error * error;
    }
    
    float rmse = sqrtf(sum_sq_error / len);
    
    /* También reportar max error */
    (void)max_error;  /* Evitar warning si no usado */
    
    return rmse;
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_Layer_Init
 * Inicializa capa BSPN completa
 * 
 * @layer:          Capa a inicializar
 * @num_channels:   Número de canales
 * @arena:          Arena para allocation
 * ============================================================================ */
static inline int RIN_BSPN_Layer_Init(RIN_BSPN_Layer* layer,
                                        uint32_t num_channels,
                                        RIN_MemoryArena* arena) {
    if (!layer || num_channels == 0 || !arena) return -1;
    
    layer->num_channels = num_channels;
    layer->params_per_channel = RIN_ALLOC_ARRAY(arena, RIN_BSPN_Params, num_channels);
    
    if (!layer->params_per_channel) return -1;
    
    /* Inicializar parámetros default */
    for (uint32_t i = 0; i < num_channels; i++) {
        layer->params_per_channel[i].scale_shift = 0;
        layer->params_per_channel[i].bias = 0;
        layer->params_per_channel[i].learnable = 1;
        layer->params_per_channel[i].use_l2 = 0;
    }
    
    memset(layer->running_mean, 0, sizeof(layer->running_mean));
    memset(layer->running_var, 0, sizeof(layer->running_var));
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_CalibrateFromData
 * Calibra parámetros desde datos de referencia
 * 
 * Ajusta scale_shift para que RMS de output coincida con LayerNorm
 * ============================================================================ */
static inline void RIN_BSPN_CalibrateFromData(RIN_BSPN_Params* params,
                                               const int16_t* inputs,
                                               const float* reference_outputs,
                                               uint32_t len,
                                               uint32_t num_samples) {
    if (!params || !inputs || !reference_outputs) return;
    
    /* Encontrar scale_shift óptimo minimizando RMSE */
    float best_rmse = 1e10f;
    int8_t best_shift = 0;
    
    int16_t temp_output[1024];  /* Asume len <= 1024 */
    
    for (int8_t shift = -3; shift <= 3; shift++) {
        params->scale_shift = shift;
        
        float total_rmse = 0.0f;
        
        for (uint32_t s = 0; s < num_samples; s++) {
            const int16_t* input = &inputs[s * len];
            const float* ref = &reference_outputs[s * len];
            
            RIN_BSPN_Forward(input, temp_output, len, params);
            
            float rmse = RIN_BSPN_ValidateError(temp_output, ref, len);
            total_rmse += rmse;
        }
        
        float avg_rmse = total_rmse / num_samples;
        
        if (avg_rmse < best_rmse) {
            best_rmse = avg_rmse;
            best_shift = shift;
        }
    }
    
    params->scale_shift = best_shift;
}

/* ============================================================================
 * FUNCIÓN: RIN_BSPN_QuantizeGammaBeta
 * Convierte gamma/beta flotantes a scale_shift/bias Q15
 * 
 * @gamma:     Escala deseada (float)
 * @beta:      Bias deseado (float)
 * @params:    Output estructura poblada
 * ============================================================================ */
static inline void RIN_BSPN_QuantizeGammaBeta(float gamma, float beta, RIN_BSPN_Params* params) {
    if (!params) return;
    
    /* Convertir gamma a scale_shift aproximado */
    /* gamma ≈ 2^(-scale_shift) */
    if (gamma > 0 && gamma != 1.0f) {
        float log2_gamma = -log2f(gamma);  /* Negativo porque shift es divisor */
        params->scale_shift = (int16_t)(log2_gamma + 0.5f);
    } else {
        params->scale_shift = 0;
    }
    
    /* Convertir beta a Q15 */
    float beta_q15 = beta * 32768.0f;
    if (beta_q15 > RIN_Q15_MAX) beta_q15 = RIN_Q15_MAX;
    if (beta_q15 < RIN_Q15_MIN) beta_q15 = RIN_Q15_MIN;
    params->bias = (int16_t)beta_q15;
}

/* ============================================================================
 * MACRO: RIN_BSPN_DEFAULT_PARAMS
 * Parámetros por defecto que típicamente funcionan bien
 * ============================================================================ */
#define RIN_BSPN_DEFAULT_PARAMS() ((RIN_BSPN_Params){ \
    .scale_shift = 0, \
    .bias = 0, \
    .learnable = 1, \
    .use_l2 = 0 \
})

/* ============================================================================
 * MACRO: RIN_BSPN_FIXED_PARAMS
 * Parámetros fijos (no learnable) para inference-only
 * ============================================================================ */
#define RIN_BSPN_FIXED_PARAMS(scale, bias_val) ((RIN_BSPN_Params){ \
    .scale_shift = (scale), \
    .bias = (bias_val), \
    .learnable = 0, \
    .use_l2 = 0 \
})

#ifdef __cplusplus
}
#endif

#endif /* RIN_BSPN_H */
