#ifndef RIN_API_H
#define RIN_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
  #define THOR_EXPORT __declspec(dllexport)
#else
  #define THOR_EXPORT __attribute__((visibility("default")))
#endif

/* Status codes */
typedef enum {
    THOR_OK = 0,
    THOR_ERR_INIT = -1,
    THOR_ERR_MEMORY = -2,
    THOR_ERR_WEIGHTS = -3,
    THOR_ERR_INFERENCE = -4,
    THOR_ERR_NOT_INITIALIZED = -5,
    THOR_ERR_INVALID_INPUT = -6,
    THOR_ERR_UNSUPPORTED = -7,
} ThorStatus;

/* Inference modes */
typedef enum {
    THOR_MODE_MLP = 0,
    THOR_MODE_SNN,
    THOR_MODE_ATTN,
    THOR_MODE_THOR,
    THOR_MODE_TRANSFORMER,
} ThorMode;

/* Model info */
typedef struct {
    uint32_t num_layers;
    uint32_t model_dim;
    uint32_t vocab_size;
    uint32_t num_heads;
    uint32_t max_seq_len;
    uint32_t ffn_dim;
    uint32_t num_parameters;
    uint32_t architecture;
    float size_mb;
} ThorModelInfo;

/* Inference result */
typedef struct {
    uint32_t* tokens;
    uint32_t num_tokens;
    double energy_joules;
    float tokens_per_second;
    uint64_t latency_ns;
} ThorResult;

/* Handle for the runtime context (opaque) */
typedef struct ThorContext ThorContext;

/* ─── Lifecycle ─────────────────────────────────────────────────────────── */

THOR_EXPORT ThorContext* thor_create(void);
THOR_EXPORT void thor_destroy(ThorContext* ctx);

/* ─── Model loading ─────────────────────────────────────────────────────── */

THOR_EXPORT ThorStatus thor_load_model(ThorContext* ctx, const char* model_path);
THOR_EXPORT ThorStatus thor_get_model_info(ThorContext* ctx, ThorModelInfo* info);

/* ─── Configuration ─────────────────────────────────────────────────────── */

THOR_EXPORT void thor_set_mode(ThorContext* ctx, ThorMode mode);
THOR_EXPORT ThorMode thor_get_mode(ThorContext* ctx);
THOR_EXPORT void thor_set_temperature(ThorContext* ctx, float temp);
THOR_EXPORT void thor_set_top_k(ThorContext* ctx, uint32_t k);
THOR_EXPORT void thor_set_top_p(ThorContext* ctx, float p);
THOR_EXPORT void thor_set_power_budget(ThorContext* ctx, float watts);

/* ─── Inference ─────────────────────────────────────────────────────────── */

THOR_EXPORT ThorStatus thor_infer(ThorContext* ctx,
                                   const uint32_t* input_ids,
                                   uint32_t num_input,
                                   uint32_t max_output,
                                   ThorResult* result);
THOR_EXPORT void thor_free_result(ThorResult* result);

/* ─── Tokenizer ─────────────────────────────────────────────────────────── */

THOR_EXPORT const char* thor_get_charset(ThorContext* ctx, int* vocab_size);
THOR_EXPORT int thor_encode(ThorContext* ctx, const char* text,
                             uint32_t* ids, int max_ids);
THOR_EXPORT void thor_decode(ThorContext* ctx, const uint32_t* ids,
                              int n, char* text, int max_text);

/* ─── Energy ────────────────────────────────────────────────────────────── */

THOR_EXPORT double thor_get_energy_joules(ThorContext* ctx);
THOR_EXPORT double thor_get_energy_millijoules(ThorContext* ctx);
THOR_EXPORT uint64_t thor_get_inference_count(ThorContext* ctx);
THOR_EXPORT uint64_t thor_get_total_tokens(ThorContext* ctx);

/* ─── Profiling ─────────────────────────────────────────────────────────── */

THOR_EXPORT ThorStatus thor_profile(ThorContext* ctx, ThorMode mode,
                                      uint32_t num_warmup, uint32_t num_iter,
                                      double* ms_per_token, double* tokens_per_sec);

/* ─── Version ───────────────────────────────────────────────────────────── */

THOR_EXPORT const char* thor_version(void);
THOR_EXPORT void thor_version_numbers(uint32_t* major, uint32_t* minor, uint32_t* patch);

#ifdef __cplusplus
}
#endif

#endif /* RIN_API_H */
