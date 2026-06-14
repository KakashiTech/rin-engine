/*
 * rin_lif_engine.h - Motor de Neuronas LIF Multiplication-Free
 * 
 * Leaky Integrate-and-Fire sin multiplicaciones de punto flotante
 * Reemplaza: H[t] = (1-τ⁻¹)·V[t-1] + τ⁻¹·X[t]
 * Con:        H[t] = (V[t-1] >> decay_shift) + (X[t] >> input_shift)
 * 
 * Basado en: NeurIPS 2025 - "Multiplication-Free Parallelizable Spiking Neurons"
 *            MINT - Multiplier-less INTeger Quantization (ASP-DAC 2024)
 */

#ifndef RIN_LIF_ENGINE_H
#define RIN_LIF_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rin_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTES Q15
 * ============================================================================ */

#define RIN_Q15_MAX     32767
#define RIN_Q15_MIN     (-32768)
#define RIN_Q15_ONE     32767   /* ~1.0 en Q15 */
#define RIN_Q15_HALF    16384   /* 0.5 en Q15 */

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_LIF_State - Estado de una neurona LIF
 * Membrane potential almacenado como entero fijo Q15
 */
typedef struct {
    int16_t v_mem;           /* Potencial de membrana (Q15) */
    int16_t threshold;       /* Umbral de disparo (Q15) */
    uint8_t decay_shift;     /* Shift para leaky: v = v >> decay_shift */
    uint8_t input_shift;     /* Shift para input: input >> input_shift */
    uint8_t reset_mode;      /* 0=zero, 1=subtract, 2=soft reset */
    bool    fired;           /* Estado de spike en este timestep */
} RIN_LIF_State;

/*
 * RIN_LIF_Layer - Capa de neuronas LIF
 */
typedef struct {
    RIN_LIF_State* neurons;       /* Array de estados */
    uint32_t       num_neurons;
    uint32_t       num_inputs;
    
    /* Pesos cuantizados - potencias de 2 para multiplicación por shift */
    int8_t*        weights;       /* log2(|w|) con signo separado */
    uint8_t*       weight_signs;  /* 0=positivo, 1=negativo (bit-packed opcional) */
    uint32_t       weight_stride;
    
    /* Estados para temporal dynamics */
    uint32_t       current_timestep;
    uint32_t       total_timesteps;
} RIN_LIF_Layer;

/*
 * RIN_LIF_Config - Configuración de neurona
 */
typedef struct {
    int16_t  threshold_q15;    /* Umbral (Q15, ej: 10000 ~ 0.3) */
    uint8_t  decay_shift;      /* 1-7, típicamente 2-4 */
    uint8_t  input_shift;      /* 1-5, típicamente 2-3 */
    uint8_t  reset_mode;       /* RIN_LIF_RESET_* */
} RIN_LIF_Config;

/* Reset modes */
#define RIN_LIF_RESET_ZERO      0   /* V = 0 después de spike */
#define RIN_LIF_RESET_SUBTRACT  1   /* V -= threshold */
#define RIN_LIF_RESET_SOFT      2   /* V *= (1 - 1/threshold) */

/* ============================================================================
 * FUNCIÓN: RIN_Q15_Saturate
 * Saturación a rango Q15
 * ============================================================================ */
static inline int16_t RIN_Q15_Saturate(int32_t value) {
    if (value > RIN_Q15_MAX) return RIN_Q15_MAX;
    if (value < RIN_Q15_MIN) return RIN_Q15_MIN;
    return (int16_t)value;
}

/* ============================================================================
 * FUNCIÓN: RIN_Q15_MulShift
 * Multiplicación Q15 usando shifts (para potencias de 2)
 * 
 * Si w es potencia de 2: x * w = x << log2(w) o x >> -log2(w)
 * 
 * @x:        Valor Q15
 * @log2_w:   log2 del peso (positivo=shift left, negativo=shift right)
 * 
 * Retorna: x * (2^log2_w) en Q15 saturado
 * ============================================================================ */
static inline int16_t RIN_Q15_MulShift(int16_t x, int8_t log2_w) {
    if (log2_w == 0) return x;
    
    int32_t result;
    if (log2_w > 0) {
        /* Multiplicación por potencia de 2 positiva */
        result = (int32_t)x << log2_w;
    } else {
        /* División (shift right) */
        result = (int32_t)x >> (-log2_w);
    }
    
    return RIN_Q15_Saturate(result);
}

/* ============================================================================
 * FUNCIÓN: RIN_Q15_AddSaturate
 * Suma con saturación Q15
 * ============================================================================ */
static inline int16_t RIN_Q15_AddSaturate(int16_t a, int16_t b) {
    return RIN_Q15_Saturate((int32_t)a + (int32_t)b);
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Init
 * Inicializa estado de neurona LIF
 * 
 * @neuron:    Estructura a inicializar
 * @config:    Parámetros de configuración
 * ============================================================================ */
static inline void RIN_LIF_Init(RIN_LIF_State* neuron, const RIN_LIF_Config* config) {
    neuron->v_mem = 0;
    neuron->threshold = config->threshold_q15;
    neuron->decay_shift = config->decay_shift;
    neuron->input_shift = config->input_shift;
    neuron->reset_mode = config->reset_mode;
    neuron->fired = false;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Update
 * Actualización de un timestep - VERSIÓN CRÍTICA
 * 
 * Implementa: v = (v >> decay) + (input >> input_shift)
 * Sin multiplicaciones de punto flotante
 * 
 * @neuron:  Estado de la neurona
 * @input:   Input Q15 (ya ponderado sumado)
 * 
 * Retorna: true si disparó spike
 * ============================================================================ */
static inline bool RIN_LIF_Update(RIN_LIF_State* neuron, int16_t input) {
    /* Paso 1: Leaky integration usando bit shift */
    int16_t leaked = neuron->v_mem >> neuron->decay_shift;
    
    /* Paso 2: Integrar input (con shift para escalar) */
    int16_t scaled_input = input >> neuron->input_shift;
    
    /* Paso 3: Sumar con saturación */
    int32_t new_v = (int32_t)leaked + (int32_t)scaled_input;
    
    /* Saturación Q15 */
    if (new_v > RIN_Q15_MAX) new_v = RIN_Q15_MAX;
    if (new_v < RIN_Q15_MIN) new_v = RIN_Q15_MIN;
    
    neuron->v_mem = (int16_t)new_v;
    
    /* Paso 4: Check umbral y reset si dispara */
    if (neuron->v_mem >= neuron->threshold) {
        /* Reset según modo */
        switch (neuron->reset_mode) {
            case RIN_LIF_RESET_ZERO:
                neuron->v_mem = 0;
                break;
            case RIN_LIF_RESET_SUBTRACT:
                neuron->v_mem = RIN_Q15_Saturate((int32_t)neuron->v_mem - neuron->threshold);
                break;
            case RIN_LIF_RESET_SOFT:
                /* Soft reset: V *= (threshold-1)/threshold */
                /* Aproximado por shift: V -= V >> log2(threshold) */
                {
                    uint8_t shift = 0;
                    int16_t tmp = neuron->threshold;
                    while (tmp > 1) { tmp >>= 1; shift++; }
                    neuron->v_mem -= (neuron->v_mem >> shift);
                }
                break;
        }
        neuron->fired = true;
        return true;  /* Spike! */
    }
    
    neuron->fired = false;
    return false;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_Init
 * Inicializa capa completa de neuronas LIF
 * 
 * @layer:         Capa a inicializar
 * @arena:         Arena para allocations
 * @num_neurons:   Número de neuronas en capa
 * @num_inputs:    Número de conexiones de entrada
 * @config:        Configuración de neurona
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
static inline int RIN_LIF_Layer_Init(RIN_LIF_Layer* layer,
                                      RIN_MemoryArena* arena,
                                      uint32_t num_neurons,
                                      uint32_t num_inputs,
                                      const RIN_LIF_Config* config) {
    if (!layer || !arena || num_neurons == 0) return -1;
    
    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;
    layer->weight_stride = num_inputs;
    layer->current_timestep = 0;
    layer->total_timesteps = 0;
    
    /* Allocar neuronas en pool de inferencia */
    layer->neurons = RIN_ALLOC_ARRAY(arena, RIN_LIF_State, num_neurons);
    if (!layer->neurons) return -1;
    
    /* Inicializar cada neurona */
    for (uint32_t i = 0; i < num_neurons; i++) {
        RIN_LIF_Init(&layer->neurons[i], config);
    }
    
    /* Allocar espacio para pesos (no inicializar aquí - se cargan después) */
    if (num_inputs > 0) {
        layer->weights = RIN_PERSIST_ALLOC(arena, int8_t, num_neurons * num_inputs);
        if (!layer->weights) return -1;
        
        /* Inicializar al valor de "sin conexión": 127 (bits 0-6 todo 1) */
        memset(layer->weights, 127, num_neurons * num_inputs * sizeof(int8_t));
    } else {
        layer->weights = NULL;
    }
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_SetWeights
 * Establece pesos cuantizados (potencias de 2)
 * 
 * @layer:      Capa objetivo
 * @log2_weights: Array de log2(|weight|) para cada conexión
 * @signs:      Array de signos (1=negativo, 0=positivo)
 * 
 * Ejemplo: peso 4.0  -> log2=2, sign=0
 *          peso -2.0 -> log2=1, sign=1
 *          peso 0.0  -> log2=-128 (código especial), o simplemente skip
 * ============================================================================ */
static inline void RIN_LIF_Layer_SetWeights(RIN_LIF_Layer* layer,
                                             const int8_t* log2_weights,
                                             const uint8_t* signs) {
    if (!layer || !layer->weights) return;
    
    uint32_t total_weights = layer->num_neurons * layer->num_inputs;
    
    for (uint32_t i = 0; i < total_weights; i++) {
        /* Codificar signo en bit 7, magnitud en bits 0-6 */
        int8_t w = log2_weights[i];
        if (signs && signs[i]) {
            w |= 0x80;  /* Set bit de signo */
        }
        layer->weights[i] = w;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_DecodeWeight
 * Decodifica peso cuantizado a valor aplicable
 * 
 * @encoded: Valor codificado (signo en bit 7, magnitud en 0-6)
 * @input:   Input a multiplicar
 * 
 * Retorna: input * weight usando solo shifts
 * ============================================================================ */
static inline int16_t RIN_LIF_DecodeAndApply(int8_t encoded, int16_t input) {
    /* Extraer signo y magnitud */
    bool negative = (encoded & 0x80) != 0;
    int8_t log2_mag = encoded & 0x7F;
    
    /* Si magnitud es 127 o -128, es "peso cero" */
    if (log2_mag == 127) return 0;
    
    /* Aplicar shift (convertir unsigned a signed) */
    int8_t log2_w = (int8_t)log2_mag;
    if (log2_w > 63) log2_w -= 128;  /* Ajustar rango */
    
    int16_t result = RIN_Q15_MulShift(input, log2_w);
    
    /* Aplicar signo */
    return negative ? -result : result;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_Forward
 * Forward pass de capa LIF completa
 * 
 * @layer:          Capa a ejecutar
 * @input_spikes:   Array binario de spikes de entrada [timesteps][num_inputs]
 * @output_spikes:  Array binario de spikes de salida [timesteps][num_neurons]
 * @timesteps:      Número de timesteps a simular
 * @timestep_input: Si true, input_spikes es [timesteps][inputs], else [inputs]
 * 
 * NOTA: Esta es la función crítica de performance
 * ============================================================================ */
static inline void RIN_LIF_Layer_Forward(RIN_LIF_Layer* layer,
                                          const uint8_t* input_spikes,
                                          uint8_t* output_spikes,
                                          uint32_t timesteps,
                                          bool timestep_input) {
    layer->total_timesteps = timesteps;
    
    for (uint32_t t = 0; t < timesteps; t++) {
        layer->current_timestep = t;
        
        const uint8_t* current_input = timestep_input 
            ? &input_spikes[t * layer->num_inputs] 
            : input_spikes;
        
        for (uint32_t n = 0; n < layer->num_neurons; n++) {
            int32_t weighted_sum = 0;
            
            /* Sumar contribuciones de inputs conectados */
            for (uint32_t i = 0; i < layer->num_inputs; i++) {
                if (current_input[i]) {
                    /* Spike presente - acumular peso */
                    int8_t w = layer->weights[n * layer->weight_stride + i];
                    
                    /* Skip si peso es cero (codificado como 127) */
                    if ((w & 0x7F) == 127) continue;
                    
                    /* Aplicar peso con shift */
                    bool negative = (w & 0x80) != 0;
                    int8_t log2_mag = w & 0x7F;
                    if (log2_mag > 15) log2_mag = (int8_t)(log2_mag - 32);
                    
                    int16_t contribution = RIN_Q15_MulShift(RIN_Q15_ONE, log2_mag);
                    weighted_sum += negative ? -contribution : contribution;
                }
            }
            
            /* Saturar y actualizar neurona */
            int16_t input_q15 = RIN_Q15_Saturate(weighted_sum);
            bool fired = RIN_LIF_Update(&layer->neurons[n], input_q15);
            
            /* Guardar output spike */
            output_spikes[t * layer->num_neurons + n] = fired ? 1 : 0;
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_Reset
 * Reset de potenciales de membrana (entre secuencias)
 * ============================================================================ */
static inline void RIN_LIF_Layer_Reset(RIN_LIF_Layer* layer) {
    if (!layer || !layer->neurons) return;
    
    for (uint32_t i = 0; i < layer->num_neurons; i++) {
        layer->neurons[i].v_mem = 0;
        layer->neurons[i].fired = false;
    }
    layer->current_timestep = 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_GetSpikeCount
 * Obtiene conteo de spikes en capa (para análisis)
 * ============================================================================ */
static inline uint32_t RIN_LIF_Layer_GetSpikeCount(const RIN_LIF_Layer* layer) {
    if (!layer || !layer->neurons) return 0;
    
    uint32_t count = 0;
    for (uint32_t i = 0; i < layer->num_neurons; i++) {
        if (layer->neurons[i].fired) count++;
    }
    return count;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_Layer_GetActivityRate
 * Tasa de actividad (0.0 - 1.0)
 * ============================================================================ */
static inline float RIN_LIF_Layer_GetActivityRate(const RIN_LIF_Layer* layer) {
    if (!layer || layer->num_neurons == 0) return 0.0f;
    return (float)RIN_LIF_Layer_GetSpikeCount(layer) / (float)layer->num_neurons;
}

/* ============================================================================
 * FUNCIÓN: RIN_LIF_QuantizeWeight
 * Cuantiza peso flotante a potencia de 2
 * 
 * @weight:   Peso en punto flotante
 * @log2_out: Output - log2 cuantizado
 * @sign_out: Output - signo (0=pos, 1=neg)
 * 
 * Retorna: error de cuantización relativo
 * ============================================================================ */
static inline float RIN_LIF_QuantizeWeight(float weight, int8_t* log2_out, uint8_t* sign_out) {
    if (weight == 0.0f) {
        *log2_out = 127;  /* Código especial para cero */
        *sign_out = 0;
        return 0.0f;
    }
    
    *sign_out = (weight < 0) ? 1 : 0;
    float abs_w = (weight < 0) ? -weight : weight;
    
    /* log2 y redondear */
    float log2_val = log2f(abs_w);
    int8_t log2_quant = (int8_t)(log2_val + 0.5f);
    
    /* Clamp a rango válido */
    if (log2_quant > 15) log2_quant = 15;
    if (log2_quant < -15) log2_quant = -15;
    
    *log2_out = log2_quant;
    
    /* Calcular error */
    float reconstructed = (log2_quant >= 0) 
        ? (float)(1 << log2_quant) 
        : 1.0f / (float)(1 << (-log2_quant));
    
    return fabsf(abs_w - reconstructed) / abs_w;
}

#ifdef __cplusplus
}
#endif

#endif /* RIN_LIF_ENGINE_H */
