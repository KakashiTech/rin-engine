/*
 * rin_core.h - RIN System Unified Main API
 * 
 * Resonant Information Nexus - Unified entry point for inference
 * 
 * Typical usage:
 *   1. RIN_Init() - Initialize context
 *   2. RIN_LoadWeights() - Load quantized model
 *   3. RIN_Inference() - Run inference
 *   4. RIN_GetMetrics() - Get metrics
 *   5. RIN_Destroy() - Cleanup
 */

#ifndef RIN_CORE_H
#define RIN_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* Public API types (RinStatus, RinMode) */
#include "rin_api.h"

/* Include all modules */
#include "rin_arena.h"
#include "rin_dptm.h"
#include "rin_lif_engine.h"
#include "rin_ptsoftmax.h"
#include "rin_bspn.h"
#include "rin_dct_engine.h"
#include "rin_phase_gating.h"
#include "rin_betti_calculator.h"
#include "rin_mechanistic_distill.h"
#include "rin_energy_meter.h"
#include "rin_test_suite.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * VERSION
 * ============================================================================ */

#define RIN_VERSION_MAJOR 1
#define RIN_VERSION_MINOR 0
#define RIN_VERSION_PATCH 0

#define RIN_VERSION_STRING "RIN v1.0.0 - Resonant Information Nexus"

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define RIN_MAX_LAYERS           64
#define RIN_MAX_SEQ_LEN          2048
#define RIN_MAX_VOCAB_SIZE       50000
#define RIN_MAX_TOKEN_ID         49999

/* ============================================================================
 * ENUMERATIONS
 * ============================================================================ */

typedef enum {
    RIN_PRECISION_Q15 = 0,
    RIN_PRECISION_Q7,
    RIN_PRECISION_INT8,
    RIN_PRECISION_FLOAT16,
} RIN_Precision;

/* ============================================================================
 * CONFIGURATION STRUCTURES
 * ============================================================================ */

/*
 * RIN_Config - Initialization configuration
 */
typedef struct {
    /* Model dimensions */
    uint32_t model_dim;          /* Model dimension (embedding) */
    uint32_t num_layers;         /* Number of LIF layers */
    uint32_t num_heads;          /* Number of attention heads */
    uint32_t vocab_size;         /* Vocabulary size */
    uint32_t max_seq_len;        /* Maximum sequence length */
    
    /* Memory */
    uint32_t arena_size_mb;      /* Arena allocator size */
    
    /* SNN */
    uint32_t timesteps;          /* Timesteps for SNN */
    int16_t  lif_threshold;      /* LIF threshold (Q15) */
    uint8_t  lif_decay_shift;    /* LIF decay shift */
    
    /* Inference mode */
    RinMode inference_mode;
    
    /* Inference configuration */
    float    temperature;        /* Temperature for sampling */
    uint32_t top_k;              /* Top-k sampling */
    float    top_p;              /* Nucleus sampling p */
    
    /* Precision */
    RIN_Precision precision;
    
    /* Phase gate */
    RIN_PhaseGate_Config phase_gate_config;
    
    /* Transformer */
    uint32_t ffn_dim;            /* FFN inner dimension (Transformer) */

    /* Debug */
    bool     enable_profiling;
    bool     enable_energy_monitoring;
    
} RIN_Config;

/*
 * RIN_Token - Token with metadata
 */
typedef struct {
    uint32_t id;                 /* Token ID in vocabulary */
    float    logit;              /* Logit that generated this token */
    float    probability;        /* Softmax probability */
    uint64_t generation_time_ns; /* Generation time */
} RIN_Token;

/*
 * RIN_Context - Inference context (full state)
 */
typedef struct {
    /* Version and state */
    uint32_t version;
    uint32_t initialized;
    RinStatus last_error;
    
    /* Memory */
    RIN_MemoryArena arena;
    
    /* Configuration (local copy) */
    RIN_Config config;
    
    /* Model layers */
    RIN_LIF_Layer layers[RIN_MAX_LAYERS];
    RIN_BSPN_Params* norm_params;
    uint32_t num_layers;
    
    /* Spectral components */
    RIN_PhaseGate_Layer* phase_gates;
    uint32_t num_phase_gates;
    
    /* Output */
    RIN_PTSoftmax_Table softmax_table;
    
    /* Metrics and monitoring */
    RIN_EnergyMeter energy_meter;
    uint64_t inference_count;
    double total_energy_joules;
    uint64_t total_tokens_generated;
    
    /* Work buffers */
    int16_t* embedding_buffer;
    int16_t* hidden_buffer;
    uint8_t* spike_buffer;
    
    /* Current sequence */
    RIN_Token sequence[RIN_MAX_SEQ_LEN];
    uint32_t seq_len;
    
    /* Internals (backends, loaded models) */
    void* _internal;
    
    /* Last generated logits (for RIN_GenerateToken) */
    int8_t last_logits[256];
    uint32_t last_logits_dim;
    
} RIN_Context;

/*
 * RIN_InferenceResult - Inference result
 */
typedef struct {
    RIN_Token* tokens;           /* Generated tokens */
    uint32_t num_tokens;         /* Number of generated tokens */
    
    /* Metrics for this inference */
    uint64_t latency_ns;         /* Total time */
    double energy_joules;        /* Energy consumed */
    float tokens_per_second;     /* Throughput */
    
    /* Internal statistics */
    float avg_sparsity;          /* Average sparsity across layers */
    uint32_t cache_hits;         /* KV cache hits (future) */
    
} RIN_InferenceResult;

/*
 * RIN_ModelStats - Model statistics
 */
typedef struct {
    uint32_t num_parameters;     /* Total number of parameters */
    uint32_t num_layers;         /* Number of layers */
    uint32_t model_dim;          /* Model dimension */
    
    /* Memory */
    size_t weights_size_bytes;   /* Weights size */
    size_t activation_size_bytes; /* Activation size */
    
    /* Quantization */
    RIN_Precision precision;
    uint8_t weight_bits;         /* Bits per weight */
    
} RIN_ModelStats;

/* ============================================================================
 * VERSION FUNCTIONS
 * ============================================================================ */

static inline const char* RIN_GetVersion(void) {
    return RIN_VERSION_STRING;
}

static inline void RIN_GetVersionNumbers(uint32_t* major, uint32_t* minor, uint32_t* patch) {
    if (major) *major = RIN_VERSION_MAJOR;
    if (minor) *minor = RIN_VERSION_MINOR;
    if (patch) *patch = RIN_VERSION_PATCH;
}

/* ============================================================================
 * CONFIGURATION FUNCTIONS
 * ============================================================================ */

/*
 * RIN_GetDefaultConfig - Gets recommended default configuration
 */
static inline RIN_Config RIN_GetDefaultConfig(void) {
    return (RIN_Config){
        .model_dim = 512,
        .num_layers = 8,
        .num_heads = 8,
        .vocab_size = 32000,
        .max_seq_len = 2048,
        .arena_size_mb = 256,
        .timesteps = 4,
        .lif_threshold = 10000,
        .lif_decay_shift = 2,
        .inference_mode = RIN_MODE_MLP,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .precision = RIN_PRECISION_Q15,
        .phase_gate_config = {2000, 50, true, 0.90f},
        .ffn_dim = 0,
        .enable_profiling = false,
        .enable_energy_monitoring = true
    };
}

/*
 * RIN_GetTinyConfig - Configuration for very limited hardware
 */
static inline RIN_Config RIN_GetTinyConfig(void) {
    return (RIN_Config){
        .model_dim = 256,
        .num_layers = 4,
        .num_heads = 4,
        .vocab_size = 16000,
        .max_seq_len = 512,
        .arena_size_mb = 64,
        .timesteps = 2,
        .lif_threshold = 8000,
        .lif_decay_shift = 1,
        .inference_mode = RIN_MODE_MLP,
        .temperature = 0.6f,
        .top_k = 20,
        .top_p = 0.95f,
        .precision = RIN_PRECISION_Q7,
        .phase_gate_config = {1500, 30, true, 0.95f},
        .enable_profiling = false,
        .enable_energy_monitoring = false
    };
}

/* ============================================================================
 * MAIN FUNCTIONS
 * ============================================================================ */

/*
 * RIN_Init - Initializes full RIN context
 * 
 * @ctx:    Context to initialize (pre-allocated by caller)
 * @config: Configuration
 * 
 * Returns: RIN_STATUS_OK on success, error code on failure
 */
static inline RinStatus RIN_Init(RIN_Context* ctx, const RIN_Config* config) {
    if (!ctx || !config) return RIN_STATUS_ERROR_INVALID_INPUT;
    
    memset(ctx, 0, sizeof(RIN_Context));
    
    /* Initialize arena */
    size_t inference_size = (size_t)config->arena_size_mb * 1024 * 1024 / 3;
    size_t scratch_size = inference_size;
    size_t persistent_size = inference_size;
    
    if (RIN_MemoryArena_Init(&ctx->arena, inference_size, scratch_size, persistent_size) != 0) {
        return RIN_STATUS_ERROR_MEMORY;
    }
    
    /* Copy configuration */
    ctx->config = *config;
    
    /* Initialize LIF layers */
    RIN_LIF_Config lif_config = {
        .threshold_q15 = config->lif_threshold,
        .decay_shift = config->lif_decay_shift,
        .input_shift = 3,
        .reset_mode = RIN_LIF_RESET_ZERO
    };
    
    for (uint32_t l = 0; l < config->num_layers && l < RIN_MAX_LAYERS; l++) {
        if (RIN_LIF_Layer_Init(&ctx->layers[l], &ctx->arena,
                                config->model_dim, config->model_dim,
                                &lif_config) != 0) {
            return RIN_STATUS_ERROR_MEMORY;
        }
    }
    ctx->num_layers = config->num_layers;
    
    /* Initialize normalization */
    ctx->norm_params = RIN_ALLOC_ARRAY(&ctx->arena, RIN_BSPN_Params, config->num_layers);
    if (!ctx->norm_params) {
        return RIN_STATUS_ERROR_MEMORY;
    }
    
    for (uint32_t l = 0; l < config->num_layers; l++) {
        ctx->norm_params[l] = RIN_BSPN_DEFAULT_PARAMS();
    }
    
    /* Initialize phase gates */
    if (config->phase_gate_config.target_sparsity > 0) {
        ctx->phase_gates = RIN_ALLOC_ARRAY(&ctx->arena, RIN_PhaseGate_Layer, config->num_layers);
        if (ctx->phase_gates) {
            for (uint32_t l = 0; l < config->num_layers; l++) {
                RIN_PhaseGate_InitLayer(&ctx->phase_gates[l], &ctx->arena,
                                        config->model_dim, config->model_dim,
                                        &config->phase_gate_config);
            }
            ctx->num_phase_gates = config->num_layers;
        }
    }
    
    /* Initialize softmax */
    RIN_PTSoftmax_InitTable(&ctx->softmax_table, 32);
    
    /* Initialize energy meter */
    if (config->enable_energy_monitoring) {
        RIN_EnergyMeter_Init(&ctx->energy_meter);
    }
    
    /* Allocate work buffers */
    ctx->embedding_buffer = RIN_ALLOC_ARRAY(&ctx->arena, int16_t, config->model_dim);
    ctx->hidden_buffer = RIN_ALLOC_ARRAY(&ctx->arena, int16_t, config->model_dim);
    ctx->spike_buffer = RIN_ALLOC_ARRAY(&ctx->arena, uint8_t, config->model_dim);
    
    if (!ctx->embedding_buffer || !ctx->hidden_buffer || !ctx->spike_buffer) {
        return RIN_STATUS_ERROR_MEMORY;
    }
    
    ctx->version = (RIN_VERSION_MAJOR << 16) | (RIN_VERSION_MINOR << 8) | RIN_VERSION_PATCH;
    ctx->initialized = 1;
    ctx->inference_count = 0;
    
    return RIN_STATUS_OK;
}

/*
 * RIN_LoadWeights - Loads quantized weights from file
 * 
 * Expected format: binary with metadata header
 * 
 * @ctx:  Initialized context
 * @path: Path to weights file
 * 
 * Returns: RIN_STATUS_OK on success
 */
RinStatus RIN_LoadWeights(RIN_Context* ctx, const char* path);

/*
 * RIN_Inference - Runs full inference
 * 
 * @ctx:         Initialized context
 * @input_ids:   Array of input token IDs
 * @num_input:   Number of input tokens
 * @max_output:  Maximum tokens to generate
 * @result:      Structure to populate with results
 * 
 * Returns: RIN_STATUS_OK on success
 */
RinStatus RIN_Inference(RIN_Context* ctx,
                          const uint32_t* input_ids,
                          uint32_t num_input,
                          uint32_t max_output,
                          RIN_InferenceResult* result);

/*
 * RIN_GenerateToken - Generates a single token
 * 
 * Low-level function for fine-grained generation control
 */
RinStatus RIN_GenerateToken(RIN_Context* ctx, RIN_Token* next_token);

/*
 * RIN_Reset - Reset for new session (keeps weights)
 * 
 * Clears neuron states and buffers, but keeps loaded weights
 */
static inline void RIN_Reset(RIN_Context* ctx) {
    if (!ctx || !ctx->initialized) return;
    
    /* Reset LIF neurons */
    for (uint32_t l = 0; l < ctx->num_layers; l++) {
        RIN_LIF_Layer_Reset(&ctx->layers[l]);
    }
    
    /* Reset phase gates */
    for (uint32_t g = 0; g < ctx->num_phase_gates; g++) {
        RIN_PhaseGate_ResetStats(&ctx->phase_gates[g]);
    }
    
    /* Reset arena buffers (keeps weights in persistent) */
    RIN_MemoryArena_ResetInference(&ctx->arena);
    
    /* Reset sequence */
    ctx->seq_len = 0;
}

/*
 * RIN_Destroy - Complete cleanup
 */
/* Internal destructor (implemented in rin_core.c) */
void RIN_Destroy_Internal(RIN_Context* ctx);

static inline void RIN_Destroy(RIN_Context* ctx) {
    if (!ctx) return;
    
    /* Close energy meter */
    if (ctx->config.enable_energy_monitoring) {
        RIN_EnergyMeter_Close(&ctx->energy_meter);
    }
    
    /* Free internal backend */
    RIN_Destroy_Internal(ctx);
    
    /* Free arena (includes all: weights, buffers, etc.) */
    RIN_MemoryArena_Destroy(&ctx->arena);
    
    ctx->initialized = 0;
}

/* ============================================================================
 * METRICS FUNCTIONS
 * ============================================================================ */

/*
 * RIN_GetModelStats - Gets model statistics
 */
static inline void RIN_GetModelStats(const RIN_Context* ctx, RIN_ModelStats* stats) {
    if (!ctx || !stats) return;
    
    stats->num_layers = ctx->num_layers;
    stats->model_dim = ctx->config.model_dim;
    stats->precision = ctx->config.precision;
    
    /* Parameter estimation */
    uint32_t params_per_layer = ctx->config.model_dim * ctx->config.model_dim;
    stats->num_parameters = params_per_layer * ctx->num_layers;
    
    /* Weights size */
    uint8_t bytes_per_param = (ctx->config.precision == RIN_PRECISION_Q15) ? 2 : 1;
    stats->weights_size_bytes = stats->num_parameters * bytes_per_param;
    
    /* Activations */
    stats->activation_size_bytes = ctx->config.model_dim * 2 * sizeof(int16_t);
}

/*
 * RIN_GetPerformanceMetrics - Gets accumulated performance metrics
 */
typedef struct {
    uint64_t total_inferences;
    uint64_t total_tokens_generated;
    double total_energy_joules;
    float avg_tokens_per_second;
    float avg_joules_per_token;
    float avg_sparsity;
    float memory_usage_ratio;
} RIN_PerformanceMetrics;

static inline void RIN_GetPerformanceMetrics(const RIN_Context* ctx,
                                            RIN_PerformanceMetrics* metrics) {
    if (!ctx || !metrics) return;
    
    memset(metrics, 0, sizeof(RIN_PerformanceMetrics));
    
    metrics->total_inferences = ctx->inference_count;
    metrics->total_tokens_generated = ctx->total_tokens_generated;
    metrics->total_energy_joules = ctx->total_energy_joules;
    
    if (ctx->inference_count > 0) {
        metrics->avg_tokens_per_second = 
            (float)ctx->total_tokens_generated / (float)ctx->inference_count;
    }
    
    if (ctx->total_tokens_generated > 0) {
        metrics->avg_joules_per_token = 
            (float)ctx->total_energy_joules / (float)ctx->total_tokens_generated;
    }
    
    metrics->memory_usage_ratio = RIN_MemPool_UsageRatio(&ctx->arena.inference);
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/*
 * RIN_PrintInfo - Prints system information
 */
static inline void RIN_PrintInfo(const RIN_Context* ctx) {
    printf("\n========== %s ==========\n", RIN_GetVersion());
    
    if (!ctx) {
        printf("Context: NULL\n");
        return;
    }
    
    printf("Initialized: %s\n", ctx->initialized ? "YES" : "NO");
    printf("Model dim: %u\n", ctx->config.model_dim);
    printf("Layers: %u\n", ctx->num_layers);
    printf("Vocab size: %u\n", ctx->config.vocab_size);
    printf("Max seq len: %u\n", ctx->config.max_seq_len);
    printf("Timesteps: %u\n", ctx->config.timesteps);
    printf("Target sparsity: %.0f%%\n", ctx->config.phase_gate_config.target_sparsity * 100.0f);
    
    if (ctx->initialized) {
        RIN_ModelStats stats;
        RIN_GetModelStats(ctx, &stats);
        printf("Parameters: ~%u (%.2f MB)\n", 
               stats.num_parameters, 
               (float)stats.weights_size_bytes / (1024.0f * 1024.0f));
    }
    
    printf("======================================\n\n");
}

/*
 * RIN_GetLastError - Gets last error
 */
static inline RinStatus RIN_GetLastError(const RIN_Context* ctx) {
    return ctx ? ctx->last_error : RIN_STATUS_ERROR_NOT_INITIALIZED;
}

/*
 * RIN_ErrorString - Converts error code to string
 */
static inline const char* RIN_ErrorString(RinStatus status) {
    switch (status) {
        case RIN_STATUS_OK: return "OK";
        case RIN_STATUS_ERROR_INIT: return "Initialization error";
        case RIN_STATUS_ERROR_MEMORY: return "Memory allocation error";
        case RIN_STATUS_ERROR_WEIGHTS: return "Weights loading error";
        case RIN_STATUS_ERROR_INFERENCE: return "Inference error";
        case RIN_STATUS_ERROR_NOT_INITIALIZED: return "Context not initialized";
        case RIN_STATUS_ERROR_INVALID_INPUT: return "Invalid input";
        default: return "Unknown error";
    }
}

/* ============================================================================
 * CONVENIENCE MACROS
 * ============================================================================ */

/* Initialize with default configuration */
#define RIN_INIT_DEFAULT(ctx) \
    RIN_Init((ctx), &RIN_GetDefaultConfig())

/* Initialize tiny configuration */
#define RIN_INIT_TINY(ctx) \
    RIN_Init((ctx), &RIN_GetTinyConfig())

/* Quick inference: one line */
#define RIN_QUICK_INFERENCE(ctx, input, n_in, output, n_out) \
    RIN_Inference((ctx), (input), (n_in), (n_out), (output))

/*
 * RIN_GetCharSet - Gets the model charset (Transformer)
 * Returns: pointer to internal buffer with vocab_size chars, or NULL if unavailable
 * The charset is stored in the .rin file after the weights
 */
const char* RIN_GetCharSet(RIN_Context* ctx, int* vocab_size_out);

#ifdef __cplusplus
}
#endif

#endif /* RIN_CORE_H */
