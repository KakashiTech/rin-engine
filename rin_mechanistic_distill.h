/*
 * rin_mechanistic_distill.h - Destilación Mecanicista
 * 
 * Replica patrones de atención internos del modelo maestro
 * No solo imita outputs - replica "frecuencias de resonancia"
 * 
 * Basado en: "Mechanistic Interpretability" + Sorbet distillation (ICML 2025)
 */

#ifndef RIN_MECHANISTIC_DISTILL_H
#define RIN_MECHANISTIC_DISTILL_H

#include <stdint.h>
#include <math.h>
#include <string.h>
#include "rin_betti_calculator.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTES
 * ============================================================================ */

#define RIN_DISTILL_MAX_ATTENTION_HEADS 16
#define RIN_DISTILL_MAX_SEQ_LEN 512
#define RIN_DISTILL_MAX_LAYERS 24

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_AttentionPattern - Patrón de atención para destilación
 * Representa "frecuencias de resonancia" del modelo maestro
 */
typedef struct {
    uint32_t num_heads;
    uint32_t seq_len;
    
    /* Patrones de atención: [heads][seq][seq] flattenado */
    float* attention_scores;     /* Softmax attention weights */
    uint16_t* top_k_indices;     /* Top-k posiciones atendidas */
    uint8_t top_k;               /* k para sparse attention */
    
    /* Métricas de resonancia */
    float resonance_freq;        /* Frecuencia dominante (FFT de atención) */
    float amplitude;             /* Fuerza de la resonancia */
    float entropy;               /* Entropía del patrón (medida de foco) */
} RIN_AttentionPattern;

/*
 * RIN_FeatureMaps - Mapas de features para matching
 */
typedef struct {
    float* values;               /* Activaciones flattenadas */
    uint32_t dim;                /* Dimensión de features */
    uint32_t count;              /* Número de vectores */
    
    /* Estadísticas para normalización */
    float mean;
    float std;
} RIN_FeatureMaps;

/*
 * RIN_DistillConfig - Configuración de destilación
 */
typedef struct {
    float alpha_ce;              /* Peso de cross-entropy */
    float alpha_kd;              /* Peso de knowledge distillation (soft targets) */
    float alpha_topo;            /* Peso de pérdida topológica */
    float alpha_attn;            /* Peso de pérdida de atención */
    float alpha_feat;            /* Peso de pérdida de features intermedias */
    float alpha_resonance;       /* Peso de pérdida de resonancia */
    float temperature;           /* Temperatura para softmax en KD */
} RIN_DistillConfig;

/*
 * RIN_DistillLossBreakdown - Desglose de pérdida de destilación
 */
typedef struct {
    float loss_ce;               /* Cross-entropy */
    float loss_kd;               /* Knowledge distillation */
    float loss_topo;             /* Pérdida topológica */
    float loss_attn;             /* Matching de atención */
    float loss_feat;             /* Matching de features */
    float loss_resonance;        /* Resonancia */
    float loss_total;            /* Total ponderado */
} RIN_DistillLossBreakdown;

/*
 * RIN_DistillState - Estado del proceso de destilación
 */
typedef struct {
    RIN_DistillConfig config;
    
    /* Referencias al maestro (no se modifican) */
    const RIN_AttentionPattern* teacher_attn;
    const RIN_FeatureMaps* teacher_features;
    const RIN_BettiNumbers* teacher_betti;
    
    /* Estado del estudiante */
    RIN_AttentionPattern student_attn;
    RIN_FeatureMaps student_features;
    RIN_BettiNumbers student_betti;
    
    /* Métricas acumuladas */
    uint32_t steps_completed;
    float avg_loss;
    float best_accuracy;
} RIN_DistillState;

/* ============================================================================
 * CONFIGURACIÓN PREDETERMINADA
 * ============================================================================ */

static inline RIN_DistillConfig RIN_Distill_GetDefaultConfig(void) {
    return (RIN_DistillConfig){
        .alpha_ce = 0.3f,
        .alpha_kd = 0.3f,
        .alpha_topo = 0.1f,
        .alpha_attn = 0.2f,
        .alpha_feat = 0.1f,
        .alpha_resonance = 0.0f,  /* Opcional, computacionalmente costoso */
        .temperature = 4.0f
    };
}

static inline RIN_DistillConfig RIN_Distill_GetAggressiveConfig(void) {
    return (RIN_DistillConfig){
        .alpha_ce = 0.1f,
        .alpha_kd = 0.2f,
        .alpha_topo = 0.3f,      /* Enfatizar topología */
        .alpha_attn = 0.3f,      /* Enfatizar atención */
        .alpha_feat = 0.1f,
        .alpha_resonance = 0.0f,
        .temperature = 8.0f
    };
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_KLDivergence
 * KL-divergence entre dos distribuciones
 * ============================================================================ */
static inline float RIN_Distill_KLDivergence(const float* p, const float* q, uint32_t len) {
    const float eps = 1e-10f;
    float kl = 0.0f;
    
    for (uint32_t i = 0; i < len; i++) {
        float pi = p[i] + eps;
        float qi = q[i] + eps;
        if (pi > eps) {
            kl += pi * logf(pi / qi);
        }
    }
    
    return kl;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_CrossEntropy
 * Cross-entropy para clasificación
 * ============================================================================ */
static inline float RIN_Distill_CrossEntropy(const float* logits,
                                            uint32_t target_class,
                                            uint32_t num_classes) {
    const float eps = 1e-10f;
    
    /* Softmax para obtener probabilidades */
    float max_logit = -FLT_MAX;
    for (uint32_t i = 0; i < num_classes; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < num_classes; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    float p_target = expf(logits[target_class] - max_logit) / sum_exp;
    
    return -logf(p_target + eps);
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_KnowledgeDistillationLoss
 * Pérdida de knowledge distillation (Hinton et al.)
 * 
 * Combinación de CE con targets duros + KL con targets suaves del maestro
 * ============================================================================ */
static inline float RIN_Distill_KnowledgeDistillationLoss(
    const float* student_logits,
    const float* teacher_logits,
    uint32_t target_class,
    uint32_t num_classes,
    float temperature,
    float alpha) {
    
    /* Cross-entropy con target duro */
    float ce = RIN_Distill_CrossEntropy(student_logits, target_class, num_classes);
    
    /* Softmax con temperatura para teacher */
    float teacher_soft[256];  /* Asume max 256 classes */
    float max_t = -FLT_MAX;
    for (uint32_t i = 0; i < num_classes; i++) {
        if (teacher_logits[i] > max_t) max_t = teacher_logits[i];
    }
    
    float sum_t = 0.0f;
    for (uint32_t i = 0; i < num_classes; i++) {
        teacher_soft[i] = expf((teacher_logits[i] - max_t) / temperature);
        sum_t += teacher_soft[i];
    }
    for (uint32_t i = 0; i < num_classes; i++) {
        teacher_soft[i] /= sum_t;
    }
    
    /* Softmax con temperatura para student */
    float student_soft[256];
    float max_s = -FLT_MAX;
    for (uint32_t i = 0; i < num_classes; i++) {
        if (student_logits[i] > max_s) max_s = student_logits[i];
    }
    
    float sum_s = 0.0f;
    for (uint32_t i = 0; i < num_classes; i++) {
        student_soft[i] = expf((student_logits[i] - max_s) / temperature);
        sum_s += student_soft[i];
    }
    for (uint32_t i = 0; i < num_classes; i++) {
        student_soft[i] /= sum_s;
    }
    
    /* KL divergence */
    float kl = RIN_Distill_KLDivergence(teacher_soft, student_soft, num_classes);
    
    /* Combinar: alpha * CE + (1-alpha) * T² * KL */
    return alpha * ce + (1.0f - alpha) * (temperature * temperature) * kl;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_AttentionMatchingLoss
 * Matching de patrones de atención
 * 
 * Penaliza diferencias en distribuciones de atención
 * ============================================================================ */
static inline float RIN_Distill_AttentionMatchingLoss(
    const RIN_AttentionPattern* teacher,
    const RIN_AttentionPattern* student) {
    
    if (!teacher || !student) return 0.0f;
    
    if (teacher->num_heads != student->num_heads ||
        teacher->seq_len != student->seq_len) {
        return 1e6f;  /* Mismatch grande si dimensiones diferentes */
    }
    
    uint32_t len = teacher->num_heads * teacher->seq_len * teacher->seq_len;
    if (len == 0) return 0.0f;
    
    float total_kl = 0.0f;
    
    for (uint32_t h = 0; h < teacher->num_heads; h++) {
        uint32_t offset = h * teacher->seq_len * teacher->seq_len;
        
        float kl = RIN_Distill_KLDivergence(
            &teacher->attention_scores[offset],
            &student->attention_scores[offset],
            teacher->seq_len * teacher->seq_len
        );
        
        total_kl += kl;
    }
    
    return total_kl / (float)teacher->num_heads;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_FeatureMatchingLoss
 * MSE entre features intermedios
 * ============================================================================ */
static inline float RIN_Distill_FeatureMatchingLoss(const RIN_FeatureMaps* teacher,
                                                  const RIN_FeatureMaps* student) {
    if (!teacher || !student) return 0.0f;
    
    if (teacher->dim != student->dim || teacher->count != student->count) {
        return 1e6f;
    }
    
    uint32_t total = teacher->dim * teacher->count;
    if (total == 0) return 0.0f;
    
    float mse = 0.0f;
    for (uint32_t i = 0; i < total; i++) {
        float diff = teacher->values[i] - student->values[i];
        mse += diff * diff;
    }
    
    return mse / (float)total;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_ResonanceLoss
 * Pérdida basada en diferencias de frecuencia de resonancia
 * 
 * Penaliza cuando estudiante no replica frecuencias dominantes del maestro
 * ============================================================================ */
static inline float RIN_Distill_ResonanceLoss(const RIN_AttentionPattern* teacher,
                                             const RIN_AttentionPattern* student) {
    if (!teacher || !student) return 0.0f;
    
    /* Diferencia de frecuencia fundamental */
    float freq_diff = fabsf(teacher->resonance_freq - student->resonance_freq);
    
    /* Diferencia de amplitud de resonancia */
    float amp_diff = fabsf(teacher->amplitude - student->amplitude);
    
    /* Penalización combinada */
    return freq_diff * freq_diff + 0.5f * amp_diff * amp_diff;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_ComputeTotalLoss
 * Pérdida completa de destilación mecanicista
 * 
 * Combina: CE + KD + Topológica + Atención + Features + Resonancia
 * ============================================================================ */
static inline RIN_DistillLossBreakdown RIN_Distill_ComputeTotalLoss(
    const float* student_logits,
    const float* teacher_logits,
    const RIN_BettiNumbers* student_betti,
    const RIN_BettiNumbers* teacher_betti,
    const RIN_AttentionPattern* teacher_attn,
    const RIN_AttentionPattern* student_attn,
    const RIN_FeatureMaps* teacher_features,
    const RIN_FeatureMaps* student_features,
    uint32_t target_class,
    uint32_t num_classes,
    const RIN_DistillConfig* config) {
    
    RIN_DistillLossBreakdown result = {0};
    
    if (!config) return result;
    
    /* Cross-entropy y KD combinados */
    result.loss_kd = RIN_Distill_KnowledgeDistillationLoss(
        student_logits, teacher_logits, target_class, num_classes,
        config->temperature, config->alpha_ce
    );
    result.loss_ce = result.loss_kd;  /* Ya incluido en KD loss */
    
    /* Pérdida topológica */
    if (config->alpha_topo > 0 && student_betti && teacher_betti) {
        result.loss_topo = RIN_Betti_TopologicalLoss(
            teacher_betti, student_betti, 1.0f
        );
    }
    
    /* Pérdida de atención */
    if (config->alpha_attn > 0 && teacher_attn && student_attn) {
        result.loss_attn = RIN_Distill_AttentionMatchingLoss(teacher_attn, student_attn);
    }
    
    /* Pérdida de features */
    if (config->alpha_feat > 0 && teacher_features && student_features) {
        result.loss_feat = RIN_Distill_FeatureMatchingLoss(teacher_features, student_features);
    }
    
    /* Pérdida de resonancia */
    if (config->alpha_resonance > 0 && teacher_attn && student_attn) {
        result.loss_resonance = RIN_Distill_ResonanceLoss(teacher_attn, student_attn);
    }
    
    /* Total ponderado */
    result.loss_total = 
        config->alpha_kd * result.loss_kd +
        config->alpha_topo * result.loss_topo +
        config->alpha_attn * result.loss_attn +
        config->alpha_feat * result.loss_feat +
        config->alpha_resonance * result.loss_resonance;
    
    return result;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_Validate
 * Verifica que estudiante hereda propiedades del maestro
 * 
 * Retorna: true si todas las métricas están dentro de tolerancia
 * ============================================================================ */
static inline bool RIN_Distill_Validate(const RIN_BettiNumbers* teacher_betti,
                                       const RIN_BettiNumbers* student_betti,
                                       const RIN_AttentionPattern* teacher_attn,
                                       const RIN_AttentionPattern* student_attn,
                                       float tolerance) {
    if (!teacher_betti || !student_betti) return false;
    
    RIN_TopologicalComparison topo = RIN_Betti_CompareRepresentations(
        teacher_betti, student_betti
    );
    
    bool topo_ok = topo.total_diff <= tolerance;
    
    bool attn_ok = true;
    if (teacher_attn && student_attn) {
        float attn_diff = fabsf(teacher_attn->entropy - student_attn->entropy);
        attn_ok = attn_diff <= tolerance;
    }
    
    return topo_ok && attn_ok;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_InitState
 * Inicializa estado de destilación
 * ============================================================================ */
static inline void RIN_Distill_InitState(RIN_DistillState* state,
                                       const RIN_DistillConfig* config) {
    if (!state || !config) return;
    
    memset(state, 0, sizeof(RIN_DistillState));
    state->config = *config;
    state->steps_completed = 0;
    state->avg_loss = 0.0f;
    state->best_accuracy = 0.0f;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_UpdateMetrics
 * Actualiza métricas acumuladas
 * ============================================================================ */
static inline void RIN_Distill_UpdateMetrics(RIN_DistillState* state,
                                            float batch_loss,
                                            float batch_accuracy) {
    if (!state) return;
    
    /* Promedio móvil */
    state->avg_loss = (state->avg_loss * state->steps_completed + batch_loss) /
                      (state->steps_completed + 1);
    
    if (batch_accuracy > state->best_accuracy) {
        state->best_accuracy = batch_accuracy;
    }
    
    state->steps_completed++;
}

/* ============================================================================
 * FUNCIÓN: RIN_Distill_PrintProgress
 * Debug: imprime progreso de destilación
 * ============================================================================ */
static inline void RIN_Distill_PrintProgress(const RIN_DistillState* state,
                                            uint32_t total_steps) {
    if (!state) return;
    
    float progress = (float)state->steps_completed / (float)total_steps * 100.0f;
    
    printf("[DISTILL] Step %u/%u (%.1f%%) - Loss: %.4f, Best Acc: %.2f%%\n",
           state->steps_completed,
           total_steps,
           progress,
           state->avg_loss,
           state->best_accuracy * 100.0f);
}

#ifdef __cplusplus
}
#endif

#endif /* RIN_MECHANISTIC_DISTILL_H */
