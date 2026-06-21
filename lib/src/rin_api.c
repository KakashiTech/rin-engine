#include "rin_api.h"
#include "rin_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Accessors for RIN_Internal fields (defined in rin_core.c) */
extern int rin_internal_num_layers(void *p);
extern int rin_internal_input_dim(void *p);
extern int rin_internal_output_dim(void *p);
extern int rin_internal_num_heads(void *p);
extern int rin_internal_ffn_dim(void *p);
extern int rin_internal_max_seq_len(void *p);

struct RinContext {
    RIN_Context ctx;
    int initialized;
    RinMode mode;
    float power_budget_watts;
};

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

RinContext* rin_create(void) {
    RinContext* tc = (RinContext*)calloc(1, sizeof(RinContext));
    if (!tc) return NULL;
    tc->initialized = 0;
    tc->mode = RIN_MODE_MLP;
    tc->power_budget_watts = 0.0f;
    return tc;
}

void rin_destroy(RinContext* tc) {
    if (!tc) return;
    if (tc->initialized) {
        RIN_Destroy(&tc->ctx);
    }
    free(tc);
}

/* ============================================================================
 * Internal: setup config from model if available
 * ============================================================================ */

static RinStatus ensure_init(RinContext* tc) {
    if (!tc) return RIN_STATUS_ERROR_INVALID_INPUT;
    if (!tc->initialized) {
        RIN_Config cfg = RIN_GetDefaultConfig();
        cfg.enable_energy_monitoring = 1;
        cfg.inference_mode = (RinMode)tc->mode;
        cfg.temperature = 0.8f;
        cfg.top_k = 40;
        cfg.top_p = 0.9f;
        if (RIN_Init(&tc->ctx, &cfg) != RIN_STATUS_OK)
            return RIN_STATUS_ERROR_INIT;
        tc->initialized = 1;
    }
    return RIN_STATUS_OK;
}

/* ============================================================================
 * Model loading
 * ============================================================================ */

RinStatus rin_load_model(RinContext* tc, const char* model_path) {
    if (!tc || !model_path) return RIN_STATUS_ERROR_INVALID_INPUT;

    RinStatus st = ensure_init(tc);
    if (st != RIN_STATUS_OK) return st;

    if (RIN_LoadWeights(&tc->ctx, model_path) != RIN_STATUS_OK)
        return RIN_STATUS_ERROR_WEIGHTS;

    /* Update context with actual model dimensions from loaded model */
    void *rin = tc->ctx._internal;
    if (rin) {
        tc->ctx.num_layers = (uint32_t)rin_internal_num_layers(rin);
        tc->ctx.config.model_dim = (uint32_t)rin_internal_input_dim(rin);
        tc->ctx.config.vocab_size = (uint32_t)rin_internal_output_dim(rin);
        tc->ctx.config.num_heads = (uint32_t)rin_internal_num_heads(rin);
        tc->ctx.config.max_seq_len = (uint32_t)rin_internal_max_seq_len(rin);
        tc->ctx.config.ffn_dim = (uint32_t)rin_internal_ffn_dim(rin);
    }

    return RIN_STATUS_OK;
}

RinStatus rin_get_model_info(RinContext* tc, RinModelInfo* info) {
    if (!tc || !info || !tc->initialized) return RIN_STATUS_ERROR_NOT_INITIALIZED;

    RIN_ModelStats stats;
    RIN_GetModelStats(&tc->ctx, &stats);

    info->num_layers = tc->ctx.num_layers;
    info->model_dim = tc->ctx.config.model_dim;
    info->vocab_size = tc->ctx.config.vocab_size;
    info->num_heads = tc->ctx.config.num_heads;
    info->max_seq_len = tc->ctx.config.max_seq_len;
    info->ffn_dim = tc->ctx.config.ffn_dim;
    info->num_parameters = stats.num_parameters;
    info->architecture = (uint32_t)(tc->ctx._internal ? 1 : 0);
    info->size_mb = (float)(stats.weights_size_bytes) / (1024.0f * 1024.0f);
    return RIN_STATUS_OK;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

void rin_set_mode(RinContext* tc, RinMode mode) {
    if (!tc) return;
    tc->mode = mode;
    if (tc->initialized)
        tc->ctx.config.inference_mode = (RinMode)mode;
}

RinMode rin_get_mode(RinContext* tc) {
    if (!tc) return RIN_MODE_MLP;
    return tc->mode;
}

void rin_set_temperature(RinContext* tc, float temp) {
    if (!tc || !tc->initialized) return;
    tc->ctx.config.temperature = temp;
}

void rin_set_top_k(RinContext* tc, uint32_t k) {
    if (!tc || !tc->initialized) return;
    tc->ctx.config.top_k = k;
}

void rin_set_top_p(RinContext* tc, float p) {
    if (!tc || !tc->initialized) return;
    tc->ctx.config.top_p = p;
}

void rin_set_power_budget(RinContext* tc, float watts) {
    if (!tc) return;
    tc->power_budget_watts = watts;
}

/* ============================================================================
 * Inference
 * ============================================================================ */

RinStatus rin_infer(RinContext* tc,
                       const uint32_t* input_ids,
                       uint32_t num_input,
                       uint32_t max_output,
                       RinResult* result) {
    if (!tc || !input_ids || !result) return RIN_STATUS_ERROR_INVALID_INPUT;
    if (!tc->initialized) return RIN_STATUS_ERROR_NOT_INITIALIZED;

    RIN_Token tokens[64];
    RIN_InferenceResult ir;
    ir.tokens = tokens;

    RinStatus st = RIN_Inference(&tc->ctx, input_ids, num_input, max_output, &ir);
    if (st != RIN_STATUS_OK) return (RinStatus)st;

    result->num_tokens = ir.num_tokens;
    result->energy_joules = ir.energy_joules;
    result->tokens_per_second = ir.tokens_per_second;
    result->latency_ns = ir.latency_ns;
    result->tokens = (uint32_t*)malloc(ir.num_tokens * sizeof(uint32_t));
    if (result->tokens) {
        for (uint32_t i = 0; i < ir.num_tokens; i++)
            result->tokens[i] = ir.tokens[i].id;
    }
    return RIN_STATUS_OK;
}

void rin_free_result(RinResult* result) {
    if (!result) return;
    free(result->tokens);
    result->tokens = NULL;
}

/* ============================================================================
 * Tokenizer
 * ============================================================================ */

const char* rin_get_charset(RinContext* tc, int* vocab_size) {
    if (!tc || !tc->initialized) return NULL;
    return RIN_GetCharSet(&tc->ctx, vocab_size);
}

int rin_encode(RinContext* tc, const char* text,
                 uint32_t* ids, int max_ids) {
    if (!tc || !text || !ids) return -1;
    int vs;
    const char* cs = RIN_GetCharSet(&tc->ctx, &vs);
    if (!cs || vs <= 0) return -1;

    int n = 0;
    for (int i = 0; text[i] && n < max_ids; i++) {
        char c = text[i];
        int found = 0;
        for (int j = 0; j < vs; j++) {
            if (cs[j] == c) { ids[n++] = (uint32_t)j; found = 1; break; }
        }
        if (!found) ids[n++] = 0;
    }
    return n;
}

void rin_decode(RinContext* tc, const uint32_t* ids,
                  int n, char* text, int max_text) {
    if (!tc || !ids || !text || max_text <= 0) return;
    int vs;
    const char* cs = RIN_GetCharSet(&tc->ctx, &vs);
    int pos = 0;
    for (int i = 0; i < n && pos < max_text - 1; i++) {
        uint32_t id = ids[i];
        if (cs && id < (uint32_t)vs)
            text[pos++] = cs[id];
        else
            text[pos++] = '?';
    }
    text[pos] = '\0';
}

/* ============================================================================
 * Energy
 * ============================================================================ */

double rin_get_energy_joules(RinContext* tc) {
    if (!tc || !tc->initialized) return 0.0;
    return tc->ctx.total_energy_joules;
}

double rin_get_energy_millijoules(RinContext* tc) {
    return rin_get_energy_joules(tc) * 1000.0;
}

uint64_t rin_get_inference_count(RinContext* tc) {
    if (!tc || !tc->initialized) return 0;
    return tc->ctx.inference_count;
}

uint64_t rin_get_total_tokens(RinContext* tc) {
    if (!tc || !tc->initialized) return 0;
    return tc->ctx.total_tokens_generated;
}

/* ============================================================================
 * Profiling
 * ============================================================================ */

RinStatus rin_profile(RinContext* tc, RinMode mode,
                          uint32_t num_warmup, uint32_t num_iter,
                          double* ms_per_token, double* tokens_per_sec) {
    if (!tc || !ms_per_token || !tokens_per_sec) return RIN_STATUS_ERROR_INVALID_INPUT;

    RinMode prev_mode = tc->mode;
    rin_set_mode(tc, mode);

    uint32_t dummy_input[4] = {0, 1, 2, 3};
    RinResult r;
    memset(&r, 0, sizeof(r));

    for (uint32_t i = 0; i < num_warmup; i++) {
        rin_infer(tc, dummy_input, 4, 1, &r);
        rin_free_result(&r);
    }

    uint64_t t0 = RIN_DPTM_GetTimestampNs();
    for (uint32_t i = 0; i < num_iter; i++) {
        rin_infer(tc, dummy_input, 4, 1, &r);
        rin_free_result(&r);
    }
    uint64_t t1 = RIN_DPTM_GetTimestampNs();

    double total_ms = (double)(t1 - t0) / 1e6;
    *ms_per_token = total_ms / (double)num_iter;
    *tokens_per_sec = 1000.0 / *ms_per_token;

    rin_set_mode(tc, prev_mode);
    return RIN_STATUS_OK;
}

/* ============================================================================
 * Version
 * ============================================================================ */

const char* rin_version(void) {
    return RIN_VERSION_STRING;
}

void rin_version_numbers(uint32_t* major, uint32_t* minor, uint32_t* patch) {
    RIN_GetVersionNumbers(major, minor, patch);
}
