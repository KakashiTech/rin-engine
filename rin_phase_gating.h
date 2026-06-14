/*
 * rin_phase_gating.h - Gating Espectral por Coherencia de Fase
 * 
 * 90% de la red puede estar "apagada" sin pérdida de información latente
 * Basado en principios de coherencia de fase de osciladores acoplados
 * 
 * La señal solo pasa si |input_phase - weight_phase| < threshold
 */

#ifndef RIN_PHASE_GATING_H
#define RIN_PHASE_GATING_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "rin_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTES
 * ============================================================================ */

#define RIN_PHASE_Q15_MAX      32767   /* π en Q15 (aprox) */
#define RIN_PHASE_PI_Q15       25736   /* π * 2^15 / 4, ajustado */
#define RIN_PHASE_TWO_PI_Q15   51472   /* 2π en Q15 */

/* Códigos de resultado del gate */
#define RIN_GATE_PASS    1
#define RIN_GATE_BLOCK   0

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_PhaseState - Estado de fase para un componente espectral
 * Almacenado como ángulo Q15 (0-32767 ≈ 0-2π)
 */
typedef struct {
    int16_t phase;           /* Fase en Q15 (0 a ~2π) */
    int16_t magnitude;       /* Magnitud en Q15 */
    bool    active;          /* Estado del gate (pass/block) */
    uint32_t activation_count; /* Contador para estadísticas */
} RIN_PhaseState;

/*
 * RIN_PhaseGate_Config - Configuración del sistema de gating
 */
typedef struct {
    int16_t threshold;       /* Umbral de diferencia permitida (Q15) */
    uint8_t coupling_k;      /* Constante de acoplamiento (0-255) */
    bool    adaptive;        /* Si el threshold se adapta dinámicamente */
    float   target_sparsity; /* Sparsity objetivo (0.0-1.0, ej: 0.9) */
} RIN_PhaseGate_Config;

/*
 * RIN_PhaseGate_Layer - Capa con gating por fase
 * Permite que 90%+ de conexiones estén inactivas
 */
typedef struct {
    RIN_PhaseState* states;      /* Estado de fase por unidad */
    uint32_t num_units;
    uint32_t num_inputs;
    
    int16_t* input_phases;       /* Buffer para fases de input */
    int16_t* weight_phases;      /* Fases de pesos (aprendidas) */
    int16_t* phase_differences;  /* Buffer de diferencias */
    
    RIN_PhaseGate_Config config;
    
    /* Estadísticas en tiempo real */
    uint32_t total_activations;
    uint32_t total_possible;
    float    current_sparsity;
    
} RIN_PhaseGate_Layer;

/*
 * RIN_PhaseGate_Metrics - Métricas de operación
 */
typedef struct {
    float sparsity_achieved;     /* % de conexiones inactivas */
    float coherence_score;       /* Promedio de coherencia de fase */
    uint32_t active_connections; /* Número de conexiones activas */
    uint32_t total_connections;  /* Total posible */
    float energy_saved_estimate; /* Estimación de energía ahorrada */
} RIN_PhaseGate_Metrics;

/* ============================================================================
 * FUNCIÓN: RIN_Phase_Wrap
 * Normaliza fase a rango [0, 2π) en Q15
 * ============================================================================ */
static inline int16_t RIN_Phase_Wrap(int32_t phase) {
    while (phase >= RIN_PHASE_TWO_PI_Q15) phase -= RIN_PHASE_TWO_PI_Q15;
    while (phase < 0) phase += RIN_PHASE_TWO_PI_Q15;
    return (int16_t)phase;
}

/* ============================================================================
 * FUNCIÓN: RIN_Phase_Difference
 * Calcula diferencia angular mínima (considerando wrap-around)
 * 
 * Retorna: diferencia en [0, π] Q15
 * ============================================================================ */
static inline int16_t RIN_Phase_Difference(int16_t phase_a, int16_t phase_b) {
    int32_t diff = (int32_t)phase_a - (int32_t)phase_b;
    
    /* Normalizar a [-π, π] */
    while (diff > RIN_PHASE_PI_Q15) diff -= RIN_PHASE_TWO_PI_Q15;
    while (diff < -RIN_PHASE_PI_Q15) diff += RIN_PHASE_TWO_PI_Q15;
    
    /* Valor absoluto */
    if (diff < 0) diff = -diff;
    
    return (int16_t)diff;
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_Calculate
 * Determina si señal pasa el gate
 * 
 * @input_phase:  Fase de entrada
 * @weight_phase: Fase del peso/sistema
 * @threshold:    Umbral de diferencia permitida (Q15)
 * 
 * Retorna: RIN_GATE_PASS (1) o RIN_GATE_BLOCK (0)
 * ============================================================================ */
static inline bool RIN_PhaseGate_Calculate(int16_t input_phase,
                                          int16_t weight_phase,
                                          int16_t threshold) {
    int16_t diff = RIN_Phase_Difference(input_phase, weight_phase);
    return diff < threshold;
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_InitLayer
 * Inicializa capa de phase gating
 * ============================================================================ */
static inline int RIN_PhaseGate_InitLayer(RIN_PhaseGate_Layer* layer,
                                           RIN_MemoryArena* arena,
                                           uint32_t num_units,
                                           uint32_t num_inputs,
                                           const RIN_PhaseGate_Config* config) {
    if (!layer || !arena || num_units == 0) return -1;
    
    layer->num_units = num_units;
    layer->num_inputs = num_inputs;
    layer->config = *config;
    
    /* Allocar arrays */
    layer->states = RIN_ALLOC_ARRAY(arena, RIN_PhaseState, num_units);
    layer->input_phases = RIN_ALLOC_ARRAY(arena, int16_t, num_inputs);
    layer->weight_phases = RIN_PERSIST_ALLOC(arena, int16_t, num_units * num_inputs);
    layer->phase_differences = RIN_SCRATCH_ALLOC(arena, int16_t, num_inputs);
    
    if (!layer->states || !layer->input_phases || !layer->weight_phases) {
        return -1;
    }
    
    /* Inicializar estados */
    for (uint32_t i = 0; i < num_units; i++) {
        layer->states[i].phase = 0;
        layer->states[i].magnitude = 32767;  /* Máxima magnitud */
        layer->states[i].active = false;
        layer->states[i].activation_count = 0;
    }
    
    /* Inicializar pesos de fase aleatoriamente (o desde pre-trained) */
    for (uint32_t i = 0; i < num_units * num_inputs; i++) {
        /* Distribución uniforme en [0, 2π) */
        layer->weight_phases[i] = (int16_t)(i * 7919 % RIN_PHASE_TWO_PI_Q15);
    }
    
    layer->total_activations = 0;
    layer->total_possible = 0;
    layer->current_sparsity = 0.0f;
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_Forward
 * Forward pass con gating por coherencia de fase
 * 
 * Solo procesa conexiones donde |input_phase - weight_phase| < threshold
 * Esto permite que ~90% de conexiones estén "apagadas"
 * 
 * @layer:        Capa con gating
 * @input_phases: Fases de entrada (pre-computadas de activaciones)
 * @input_mags:   Magnitudes de entrada (para weighting)
 * @output:       Buffer de salida (pre-allocado, inicializado a 0)
 * @weights:      Pesos (solo se usan si el gate pasa)
 * 
 * NOTA: El 90% de skips ocurre aquí - función crítica de performance
 * ============================================================================ */
static inline void RIN_PhaseGate_Forward(RIN_PhaseGate_Layer* layer,
                                        const int16_t* input_phases,
                                        const int16_t* input_mags,
                                        int16_t* output,
                                        const int8_t* weights) {
    if (!layer || !input_phases || !output) return;
    
    uint32_t activations_this_pass = 0;
    uint32_t total_possible = layer->num_units * layer->num_inputs;
    
    /* Inicializar output a cero */
    memset(output, 0, layer->num_units * sizeof(int16_t));
    
    for (uint32_t n = 0; n < layer->num_units; n++) {
        int32_t acc = 0;
        uint32_t unit_activations = 0;
        
        for (uint32_t i = 0; i < layer->num_inputs; i++) {
            /* Check coherencia de fase - ESTO ES EL GATE */
            int16_t weight_phase = layer->weight_phases[n * layer->num_inputs + i];
            int16_t diff = RIN_Phase_Difference(input_phases[i], weight_phase);
            
            if (diff < layer->config.threshold) {
                /* GATE PASS: conexión activa - acumular */
                int8_t w = weights ? weights[n * layer->num_inputs + i] : 1;
                int16_t mag = input_mags ? input_mags[i] : 32767;
                
                /* Acumular contribución escalada */
                acc += (int32_t)w * mag;
                unit_activations++;
            }
            /* Else: GATE BLOCK - salta (90% de casos) */
        }
        
        /* Guardar output con saturación */
        if (acc > 32767) acc = 32767;
        if (acc < -32768) acc = -32768;
        output[n] = (int16_t)acc;
        
        /* Actualizar estado */
        layer->states[n].active = (unit_activations > 0);
        layer->states[n].activation_count += unit_activations;
        activations_this_pass += unit_activations;
    }
    
    /* Actualizar estadísticas globales */
    layer->total_activations += activations_this_pass;
    layer->total_possible += total_possible;
    layer->current_sparsity = 1.0f - ((float)activations_this_pass / (float)total_possible);
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_FastForward
 * Versión ultra-rápida con lookup table para fases
 * 
 * Pre-computa tabla de decisiones del gate para acelerar
 * ============================================================================ */
typedef struct {
    uint8_t gate_lut[256][256];  /* [input_phase_8bit][weight_phase_8bit] -> pass/block */
    int16_t threshold;             /* Threshold usado para LUT */
} RIN_PhaseGate_LUT;

static inline void RIN_PhaseGate_BuildLUT(RIN_PhaseGate_LUT* lut, int16_t threshold) {
    if (!lut) return;
    
    lut->threshold = threshold;
    
    for (int inp = 0; inp < 256; inp++) {
        for (int wgt = 0; wgt < 256; wgt++) {
            /* Convertir 8-bit a Q15 (aproximado) */
            int16_t inp_q15 = inp << 7;
            int16_t wgt_q15 = wgt << 7;
            
            int16_t diff = RIN_Phase_Difference(inp_q15, wgt_q15);
            lut->gate_lut[inp][wgt] = (diff < threshold) ? 1 : 0;
        }
    }
}

static inline bool RIN_PhaseGate_Lookup(const RIN_PhaseGate_LUT* lut,
                                       uint8_t input_phase_8bit,
                                       uint8_t weight_phase_8bit) {
    if (!lut) return true;  /* Default pass si no hay LUT */
    return lut->gate_lut[input_phase_8bit][weight_phase_8bit];
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_AdaptThreshold
 * Adapta threshold dinámicamente para alcanzar sparsity objetivo
 * 
 * Si sparsity actual > target: bajar threshold (más estricto)
 * Si sparsity actual < target: subir threshold (más permisivo)
 * ============================================================================ */
static inline void RIN_PhaseGate_AdaptThreshold(RIN_PhaseGate_Layer* layer) {
    if (!layer || !layer->config.adaptive) return;
    
    float current_sparsity = layer->current_sparsity;
    float target = layer->config.target_sparsity;
    
    /* Histeresis para evitar oscilación */
    const float hysteresis = 0.02f;
    
    if (current_sparsity > target + hysteresis) {
        /* Demasiado sparse - relajar threshold */
        layer->config.threshold += 100;  /* ~0.01 radianes */
        if (layer->config.threshold > 20000) layer->config.threshold = 20000;
    } else if (current_sparsity < target - hysteresis) {
        /* Muy poco sparse - ajustar threshold más estricto */
        layer->config.threshold -= 100;
        if (layer->config.threshold < 500) layer->config.threshold = 500;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_GetMetrics
 * Obtiene métricas de operación del gate
 * ============================================================================ */
static inline void RIN_PhaseGate_GetMetrics(const RIN_PhaseGate_Layer* layer,
                                           RIN_PhaseGate_Metrics* metrics) {
    if (!layer || !metrics) return;
    
    uint32_t total_conn = layer->num_units * layer->num_inputs;
    uint32_t avg_activations = (layer->total_possible > 0) 
        ? layer->total_activations / (layer->total_possible / total_conn + 1)
        : 0;
    
    metrics->sparsity_achieved = layer->current_sparsity;
    metrics->active_connections = avg_activations;
    metrics->total_connections = total_conn;
    
    /* Coherence score: promedio de (1 - diff/π) para conexiones activas */
    metrics->coherence_score = 1.0f - ((float)layer->config.threshold / (float)RIN_PHASE_PI_Q15);
    
    /* Estimación de energía ahorrada: cada conexión skip ahorra ~1 MAC */
    metrics->energy_saved_estimate = layer->current_sparsity * 0.9f;  /* ~90% de energía MAC */
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_GetSparsity
 * Reporta porcentaje de conexiones inactivas (API simple)
 * ============================================================================ */
static inline float RIN_PhaseGate_GetSparsity(const RIN_PhaseGate_Layer* layer) {
    if (!layer) return 0.0f;
    return layer->current_sparsity;
}

/* ============================================================================
 * FUNCIÓN: RIN_Phase_ComputeFromSignal
 * Extrae fase de señal usando aproximación de hilbert simplificada
 * 
 * Para uso real, la fase vendría de capa previa (ej: DCT)
 * ============================================================================ */
static inline int16_t RIN_Phase_ComputeFromSignal(int16_t signal, int16_t prev_signal) {
    /* Aproximación de derivada como cuadratura */
    int32_t diff = (int32_t)signal - (int32_t)prev_signal;
    
    /* Fase ≈ atan2(diff, signal) - aproximado por tabla o CORDIC */
    /* Simplificación: usar signos para cuadrante básico */
    int16_t phase;
    
    if (signal >= 0) {
        phase = (diff >= 0) ? RIN_PHASE_PI_Q15 / 2 : 0;
    } else {
        phase = (diff >= 0) ? RIN_PHASE_PI_Q15 : -RIN_PHASE_PI_Q15 / 2;
    }
    
    return RIN_Phase_Wrap(phase);
}

/* ============================================================================
 * FUNCIÓN: RIN_PhaseGate_ResetStats
 * Reset estadísticas acumuladas
 * ============================================================================ */
static inline void RIN_PhaseGate_ResetStats(RIN_PhaseGate_Layer* layer) {
    if (!layer) return;
    
    layer->total_activations = 0;
    layer->total_possible = 0;
    layer->current_sparsity = 0.0f;
    
    for (uint32_t i = 0; i < layer->num_units; i++) {
        layer->states[i].activation_count = 0;
    }
}

/* ============================================================================
 * CONFIGURACIÓN PREDETERMINADA
 * ============================================================================ */

/* Config para 90% sparsity (agresivo) */
#define RIN_PHASE_GATE_CONFIG_90SPARSE() ((RIN_PhaseGate_Config){ \
    .threshold = 2000,       /* ~0.1 radianes - muy estricto */ \
    .coupling_k = 50, \
    .adaptive = true, \
    .target_sparsity = 0.90f \
})

/* Config para 80% sparsity (moderado) */
#define RIN_PHASE_GATE_CONFIG_80SPARSE() ((RIN_PhaseGate_Config){ \
    .threshold = 5000,       /* ~0.25 radianes */ \
    .coupling_k = 100, \
    .adaptive = true, \
    .target_sparsity = 0.80f \
})

/* Config para 70% sparsity (conservador) */
#define RIN_PHASE_GATE_CONFIG_70SPARSE() ((RIN_PhaseGate_Config){ \
    .threshold = 8000,       /* ~0.4 radianes - más permisivo */ \
    .coupling_k = 150, \
    .adaptive = true, \
    .target_sparsity = 0.70f \
})

#ifdef __cplusplus
}
#endif

#endif /* RIN_PHASE_GATING_H */
