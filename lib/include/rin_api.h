#ifndef RIN_API_H
#define RIN_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
  #define RIN_EXPORT __declspec(dllexport)
#else
  #define RIN_EXPORT __attribute__((visibility("default")))
#endif

/* Status codes */
typedef enum {
    RIN_STATUS_OK = 0,
    RIN_STATUS_ERROR_INIT = -1,
    RIN_STATUS_ERROR_MEMORY = -2,
    RIN_STATUS_ERROR_WEIGHTS = -3,
    RIN_STATUS_ERROR_INFERENCE = -4,
    RIN_STATUS_ERROR_NOT_INITIALIZED = -5,
    RIN_STATUS_ERROR_INVALID_INPUT = -6,
    RIN_STATUS_ERROR_UNSUPPORTED = -7,
} RinStatus;

/* Inference modes */
typedef enum {
    RIN_MODE_MLP = 0,
    RIN_MODE_SNN,
    RIN_MODE_ATTN,
    RIN_MODE_THOR,
    RIN_MODE_TRANSFORMER,
} RinMode;

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
} RinModelInfo;

/* Inference result */
typedef struct {
    uint32_t* tokens;
    uint32_t num_tokens;
    double energy_joules;
    float tokens_per_second;
    uint64_t latency_ns;
} RinResult;

/* Handle for the runtime context (opaque) */
typedef struct RinContext RinContext;

/* ─── Lifecycle ─────────────────────────────────────────────────────────── */

RIN_EXPORT RinContext* rin_create(void);
RIN_EXPORT void rin_destroy(RinContext* ctx);

/* ─── Model loading ─────────────────────────────────────────────────────── */

RIN_EXPORT RinStatus rin_load_model(RinContext* ctx, const char* model_path);
RIN_EXPORT RinStatus rin_get_model_info(RinContext* ctx, RinModelInfo* info);

/* ─── Configuration ─────────────────────────────────────────────────────── */

RIN_EXPORT void rin_set_mode(RinContext* ctx, RinMode mode);
RIN_EXPORT RinMode rin_get_mode(RinContext* ctx);
RIN_EXPORT void rin_set_temperature(RinContext* ctx, float temp);
RIN_EXPORT void rin_set_top_k(RinContext* ctx, uint32_t k);
RIN_EXPORT void rin_set_top_p(RinContext* ctx, float p);
RIN_EXPORT void rin_set_power_budget(RinContext* ctx, float watts);

/* ─── Inference ─────────────────────────────────────────────────────────── */

RIN_EXPORT RinStatus rin_infer(RinContext* ctx,
                                   const uint32_t* input_ids,
                                   uint32_t num_input,
                                   uint32_t max_output,
                                   RinResult* result);
RIN_EXPORT void rin_free_result(RinResult* result);

/* ─── Tokenizer ─────────────────────────────────────────────────────────── */

RIN_EXPORT const char* rin_get_charset(RinContext* ctx, int* vocab_size);
RIN_EXPORT int rin_encode(RinContext* ctx, const char* text,
                             uint32_t* ids, int max_ids);
RIN_EXPORT void rin_decode(RinContext* ctx, const uint32_t* ids,
                              int n, char* text, int max_text);

/* ─── Energy ────────────────────────────────────────────────────────────── */

RIN_EXPORT double rin_get_energy_joules(RinContext* ctx);
RIN_EXPORT double rin_get_energy_millijoules(RinContext* ctx);
RIN_EXPORT uint64_t rin_get_inference_count(RinContext* ctx);
RIN_EXPORT uint64_t rin_get_total_tokens(RinContext* ctx);

/* ─── Profiling ─────────────────────────────────────────────────────────── */

RIN_EXPORT RinStatus rin_profile(RinContext* ctx, RinMode mode,
                                      uint32_t num_warmup, uint32_t num_iter,
                                      double* ms_per_token, double* tokens_per_sec);

/* ─── Version ───────────────────────────────────────────────────────────── */

RIN_EXPORT const char* rin_version(void);
RIN_EXPORT void rin_version_numbers(uint32_t* major, uint32_t* minor, uint32_t* patch);

#ifdef __cplusplus
}
#endif

#endif /* RIN_API_H */
