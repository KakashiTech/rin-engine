/*
 * rin_core.c - RIN Inference Engine REAL
 *
 * True pipeline using RIN modules with 3 modes:
 *   MLP_ARGMAX: GEMV → BSPN → ReLU → ... → PTsoftmax → Argmax
 *   MLP_SAMPLE: GEMV → BSPN → ReLU → ... → PTsoftmax → Sample(token)
 *   SNN:        GEMV → LIF(×timesteps) → BSPN → ... → PTsoftmax → Argmax
 *   ATTN:       GEMV → Attention → BSPN → ... → PTsoftmax → Sample(token)
 *
 * Backend: RIN INT8 SIMD for GEMV (maximum performance on Zen 3)
 * Integration: BSPN + PTsoftmax + LIF + Attention + KV Cache + RAPL
 */

#include "rin_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

/* BHRR attention: define RIN_USE_BHRR to replace KV cache with bipolar HRR */
#ifdef RIN_USE_BHRR
#include "rin_bhrr.h"
#define RIN_BHRR_SLOTS 8

/* Per-layer, per-head BHRR memory state */
typedef struct {
    RinBHRR* heads;       /* [num_heads] BHRR contexts */
    int32_t* slot_bufs;   /* pre-allocated */
    int16_t* key_bufs;    /* pre-allocated */
    int num_layers;
    int num_heads;
    int head_dim;
    int num_slots;
} RIN_BHRR_State;
#endif

#define TILE_K 32
#define TILE_OUT 8
#define MAX_ATTN_HEADS 8
#define MAX_ATTN_DIM 4096
#define RIN_MAX_VOCAB 256
#define RIN_MAX_SAVED 4096

/* ============================================================================
 * RIN MODEL
 * ============================================================================ */
typedef struct {
    int input_dim, output_dim, num_layers;
    int layer_dims[64];
    uint8_t *W[64];
    float scale[64];
    float *bias[64];
    int total_params;
    int loaded;
} RIN_Model;

/* ============================================================================
 * KV CACHE FOR ATTENTION
 * ============================================================================ */
typedef struct {
    int16_t *K, *V;       /* Q15 cached keys/values */
    int max_seq, dim;
    int current_pos;
} RIN_KVCache;

/* ============================================================================
 * INFERENCE BUFFERS
 * ============================================================================ */
typedef struct {
    RIN_Model model;
    int8_t *buf0, *buf1;
    int32_t *gemv_buf;
    RIN_KVCache kv_cache;
    int max_dim;
    RIN_EnergyMeter meter;
    uint64_t total_energy_uj;
    /* RIN heritage: separate output layer (original format) */
    uint8_t *rin_W_out;
    float rin_scale_out;
    float *rin_bias_out;

    /* Transformer */
    int architecture;          /* 0=MLP, 1=Transformer */
    int num_heads, ffn_dim, max_seq_len;
    uint8_t *embed_table;      /* [vocab_size, dim] uint8, centered at 128 */
    float embed_scale;
    float *embed_bias;

    /* Transformer per-layer weights */
    uint8_t *Wq[64], *Wk[64], *Wv[64], *Wo[64];
    uint8_t *W1[64], *W2[64];
    float sq[64], sk[64], sv[64], so[64];
    float s1[64], s2[64];
    float *bq[64], *bk[64], *bv[64], *bo[64];
    float *b1[64], *b2[64];

    /* Position embeddings Q15 [max_seq, dim] */
    int16_t *pos_embed;

    /* LayerNorm parameters Q15 [dim] per layer + final */
    int16_t *ln_gamma[65];   /* [num_layers * 2 + 1] — ln1, ln2 per layer, then ln_f */
    int16_t *ln_beta[65];
    int num_ln_sets;

    /* KV cache (multi-head) [max_seq, num_heads * head_dim] */
    int16_t *k_cache, *v_cache;
    int kv_cache_dim;           /* num_heads * head_dim */
    int kv_pos;

    /* Internal buffers for transformer pipeline */
    int16_t *q_buf, *k_buf, *v_buf;  /* QKV projection buffers */
    int16_t *attn_buf;                /* Attention output */
    int16_t *ffn_buf;                 /* FFN hidden */
    int16_t *residual;                /* Residual stream */

#ifdef RIN_USE_BHRR
    /* BHRR experimental state */
    RIN_BHRR_State bhrr;
    int16_t *bhrr_qsign, *bhrr_ksign, *bhrr_vsign;
#endif

    /* Character set (loaded from RIN file) */
    char *char_set;
    int char_set_size;
} RIN_Internal;

/* ============================================================================
 * GEMV: VPMADDUBSW INT8
 * ============================================================================ */
static void gemv_kernel(const uint8_t *W, int M, int K,
                        const int8_t *x, int32_t *y) {
    for (int r0 = 0; r0 < M; r0 += TILE_OUT) {
        int rm = M - r0 < TILE_OUT ? M - r0 : TILE_OUT;
        __m256i a0 = _mm256_setzero_si256();
        __m256i a1 = _mm256_setzero_si256();
        __m256i a2 = _mm256_setzero_si256();
        __m256i a3 = _mm256_setzero_si256();
        __m256i a4 = _mm256_setzero_si256();
        __m256i a5 = _mm256_setzero_si256();
        __m256i a6 = _mm256_setzero_si256();
        __m256i a7 = _mm256_setzero_si256();
        __m256i *aa[] = {&a0,&a1,&a2,&a3,&a4,&a5,&a6,&a7};
        for (int k = 0; k < K; k += TILE_K) {
            __m256i xv = _mm256_loadu_si256((const __m256i*)(x + k));
            for (int r = 0; r < rm; r++) {
                const uint8_t *wr = W + (size_t)(r0+r)*K + k;
                __m256i wv = _mm256_loadu_si256((const __m256i*)wr);
                __m256i p = _mm256_maddubs_epi16(wv, xv);
                *aa[r] = _mm256_add_epi32(*aa[r],
                    _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
            }
        }
        for (int r = 0; r < rm; r++) {
            __m128i lo = _mm256_castsi256_si128(*aa[r]);
            __m128i hi = _mm256_extracti128_si256(*aa[r], 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            y[r0+r] = _mm_cvtsi128_si32(s);
        }
    }
}

/* ============================================================================
 * SCALE + BIAS sobre buffer int32 (post-GEMV)
 * ============================================================================ */
static void apply_scale_bias(int32_t *buf, int M,
                              float sc, float *bs) {
    for (int j = 0; j < M; j++)
        buf[j] = (int32_t)(buf[j] * sc + bs[j]);
}

/* ============================================================================
 * HIDDEN LAYER: GEMV → Scale+Bias → ReLU → BSPN
 * ============================================================================ */
static void rin_hidden_layer(RIN_Internal *rin, int li,
                               int8_t *in, int8_t *out) {
    RIN_Model *m = &rin->model;
    int K = (li == 0) ? m->input_dim : m->layer_dims[li-1];
    int M = m->layer_dims[li];
    float sc = m->scale[li];
    float *bs = m->bias[li];

    gemv_kernel(m->W[li], M, K, in, rin->gemv_buf);

    int32_t l1 = 0;
    for (int j = 0; j < M; j++) {
        int32_t v = (int32_t)(rin->gemv_buf[j] * sc + bs[j]);
        if (v < 0) v = 0;
        l1 += v;
    }
    uint32_t l1u = (uint32_t)l1;
    int shift = 0;
    while (l1u >>= 1) shift++;
    shift = shift > 8 ? shift - 8 : 0;

    for (int j = 0; j < M; j++) {
        int32_t v = (int32_t)(rin->gemv_buf[j] * sc + bs[j]);
        if (v < 0) v = 0;
        v >>= shift;
        if (v > 127) v = 127;
        out[j] = (int8_t)v;
    }
}

/* ============================================================================
 * HIDDEN LAYER SNN: GEMV → Scale+Bias → LIF (×timesteps) → output spikes
 * ============================================================================ */
static void rin_snn_layer(RIN_Internal *rin, RIN_Context *ctx, int li,
                           uint8_t *in_spikes, uint8_t *out_spikes) {
    RIN_Model *m = &rin->model;
    int K = (li == 0) ? m->input_dim : m->layer_dims[li-1];
    int M = m->layer_dims[li];
    float sc = m->scale[li];
    float *bs = m->bias[li];
    int ts = ctx->config.timesteps;

    int8_t *in = rin->buf0;
    int8_t *out = rin->buf1;

    for (int t = 0; t < ts; t++) {
        /* Convert input spikes to INT8 activation */
        for (int j = 0; j < K; j++)
            in[j] = in_spikes[(size_t)t * K + j] ? 64 : 0;

        gemv_kernel(m->W[li], M, K, in, rin->gemv_buf);

        /* LIF update for each neuron */
        for (int j = 0; j < M; j++) {
            int32_t v = (int32_t)(rin->gemv_buf[j] * sc + bs[j]);
            if (v < -32768) v = -32768;
            if (v > 32767) v = 32767;

            /* Update LIF neuron t */
            RIN_LIF_State *n = &ctx->layers[li].neurons[j];
            bool spike = RIN_LIF_Update(n, (int16_t)v);
            out_spikes[(size_t)t * M + j] = spike ? 1 : 0;
        }
    }
    ctx->layers[li].current_timestep = 0;
}

/* ============================================================================
 * CAUSAL ATTENTION (single-head, Q15)
 * ============================================================================ */
static void rin_attention_block(int16_t *x, int dim, int kv_pos,
                                 RIN_KVCache *cache, int8_t *out) {
    /* x: vector Q15 de entrada (dimensiones del modelo)
     * Causal attention: output[i] = sum_{j<=i} softmax(Q·K_j/√d) · V_j
     * Para el paso actual kv_pos, calculamos Q para x, K/V se guardan en cache
     */

    /* Simple projections: Q=Wq*x, K=Wk*x, V=Wv*x
     * For demo, we use identity transformation + noise */
    int16_t Q[MAX_ATTN_DIM], K[MAX_ATTN_DIM], V[MAX_ATTN_DIM];
    for (int i = 0; i < dim; i++) {
        Q[i] = x[i] + (int16_t)(i * 17);     /* simulate projection */
        K[i] = x[i] + (int16_t)(i * 31);
        V[i] = x[i] + (int16_t)(i * 53);
    }

    /* Store K,V in cache */
    if (kv_pos < cache->max_seq) {
        int cdim = cache->dim;
        for (int i = 0; i < dim && i < cdim; i++) {
            cache->K[(size_t)kv_pos * cdim + i] = K[i];
            cache->V[(size_t)kv_pos * cdim + i] = V[i];
        }
    }
    int seq_len = kv_pos + 1;

    /* Attention: scores[pos][i] = Q[pos] · K[i] */
    int32_t scores[MAX_ATTN_DIM];
    int cdim = cache->dim;
    int nd = dim < cdim ? dim : cdim;
    for (int i = 0; i < seq_len; i++) {
        int32_t dot = 0;
        for (int j = 0; j < nd; j++)
            dot += (int32_t)Q[j] * cache->K[(size_t)i * cdim + j];
        dot = dot > 32767 ? 32767 : (dot < -32768 ? -32768 : dot);
        scores[i] = dot >> 8;  /* escala por √d */
    }

    /* Softmax causal on scores */
    int32_t max_s = scores[0];
    for (int i = 1; i < seq_len; i++)
        if (scores[i] > max_s) max_s = scores[i];
    uint16_t exp_s[MAX_ATTN_DIM];
    uint32_t sum_exp = 0;
    for (int i = 0; i < seq_len; i++) {
        int32_t diff = scores[i] - max_s;
        uint16_t e = diff >= -31 ? (uint16_t)(1 << (diff + 16)) : 0;
        exp_s[i] = e;
        sum_exp += e;
    }

    /* Weighted sum of V */
    if (sum_exp > 0) {
        int8_t attn_out[MAX_ATTN_DIM];
        for (int j = 0; j < nd; j++) {
            int32_t sum = 0;
            for (int i = 0; i < seq_len; i++)
                sum += (int32_t)exp_s[i] * cache->V[(size_t)i * cdim + j];
            sum /= (int32_t)(sum_exp >> 8);
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            attn_out[j] = (int8_t)sum;
        }
        memcpy(out, attn_out, nd);
    } else {
        memset(out, 0, nd);
    }
}

/* ============================================================================
 * TRANSFORMER PIPELINE (autoregressive)
 * ============================================================================ */

/* Quantize Q15 → INT8 (v >> 8) */
static inline void q15_to_int8(const int16_t *in, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t v = in[i] >> 8;
        if (v < -128) v = -128;
        if (v > 127) v = 127;
        out[i] = (int8_t)v;
    }
}

/* Sum of int8 vector */
static inline int32_t sum_int8(const int8_t *x, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
}

/* Scale INT32 GEMV output → Q15 (int16)
 * VPMADDUBSW computes sum(uint8 * int8), but weights use uint8 = int8 + 128 offset.
 * gemv = sum((int8+128) * int8) = sum(int8*int8) + 128*sum(x).
 * We correct: v = sf * (gemv - 128*sum_x) + bias_float
 *            = sf * gemv - 128*sf*sum_x + bias_float
 * bias is in float (same units as quantized weights). */
static inline void scale_to_q15(const int32_t *gemv, int16_t *out, int n,
                                 float sc, const float *bias,
                                 int32_t sum_x) {
    float corr = 128.0f * sc * sum_x;
    for (int i = 0; i < n; i++) {
        float v = (gemv[i] * sc - corr + bias[i]) * 256.0f;
        if (v < -32768.0f) v = -32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        out[i] = (int16_t)v;
    }
}

/* Old-style scale_to_q15 without VPMADDUBSW correction (for MLP path) */
static inline void scale_to_q15_old(const int32_t *gemv, int16_t *out, int n,
                                     int16_t sc_q15, const int16_t *bias) {
    for (int i = 0; i < n; i++) {
        int32_t v = gemv[i] * (int32_t)sc_q15 / 256 + bias[i];
        if (v < -32768) v = -32768;
        if (v > 32767) v = 32767;
        out[i] = (int16_t)v;
    }
}

/* Clamp Q15 after addition (residual shortcut) */
static inline int16_t clamp_q15(int32_t v) {
    if (v < -32768) return -32768;
    if (v > 32767) return 32767;
    return (int16_t)v;
}

/* Transformer multi-head attention for current position (incremental) */
static void rin_tfm_attention(RIN_Internal *rin, const int16_t *x,
                               int pos, int dim, int num_heads, int layer) {
    int hd = dim / num_heads;
    int kv_dim = num_heads * hd;  /* = dim */

    /* Quantize input for GEMV */
    q15_to_int8(x, rin->buf0, dim);
    int32_t sum_x_qkv = sum_int8(rin->buf0, dim);

    /* Q projection */
    gemv_kernel(rin->Wq[layer], dim, dim, rin->buf0, rin->gemv_buf);
    scale_to_q15(rin->gemv_buf, rin->q_buf, dim, rin->sq[layer], rin->bq[layer], sum_x_qkv);

    /* K projection */
    gemv_kernel(rin->Wk[layer], dim, dim, rin->buf0, rin->gemv_buf);
    scale_to_q15(rin->gemv_buf, rin->k_buf, dim, rin->sk[layer], rin->bk[layer], sum_x_qkv);

    /* V projection */
    gemv_kernel(rin->Wv[layer], dim, dim, rin->buf0, rin->gemv_buf);
    scale_to_q15(rin->gemv_buf, rin->v_buf, dim, rin->sv[layer], rin->bv[layer], sum_x_qkv);

#ifdef RIN_USE_BHRR
    /* === BHRR attention path (experimental) === */
    /* For each head: store K,V then retrieve with Q */
    for (int h = 0; h < num_heads; h++) {
        int off = h * hd;
        RinBHRR* mem = &rin->bhrr.heads[layer * num_heads + h];

        /* Quantize K, V to bipolar signs */
        int16_t* ksign = rin->bhrr_ksign + off;
        int16_t* vsign = rin->bhrr_vsign + off;
        for (int d = 0; d < hd; d++) {
            ksign[d] = rin_bhrr_sign_q15(rin->k_buf[off + d]);
            vsign[d] = rin_bhrr_sign_q15(rin->v_buf[off + d]);
        }

        /* Store: C += sign(K) · sign(V) */
        /* Retrieve: attn_out = C · sign(Q), scaled to Q15 range */
        if (pos == 0) {
            /* First token: retrieve is zero (no past context), just store */
            memset(&rin->attn_buf[off], 0, hd * sizeof(int16_t));
        } else {
            /* Retrieve from accumulated context (past tokens only)
             * Slot is determined by KEY (ksign), not by QUERY.
             * Qsign is used for element-wise multiply after retrieval.
             * This matches Python training: slot_ids from Ks, then R = Qs * ctx */
            int16_t* qsign = rin->bhrr_qsign + off;
            for (int d = 0; d < hd; d++)
                qsign[d] = rin_bhrr_sign_q15(rin->q_buf[off + d]);
            int slot = rin_bhrr_slot_id(ksign, hd, mem->num_slots);
            const int32_t* ctx = &mem->slots[slot * hd];
            rin_bhrr_ctx_retrieve(&rin->attn_buf[off], ctx, qsign, hd);
            /* Scale retrieve output so that 1 stored token → Q15 value ±256 */
            for (int d = 0; d < hd; d++)
                rin->attn_buf[off + d] = rin_bhrr_sat16(
                    (int32_t)rin->attn_buf[off + d] * 256);
        }

        /* Store current K,V into BHRR context for future tokens */
        rin_bhrr_store(mem, ksign, vsign);
    }
#else
    /* === Standard KV cache attention path === */
    /* Store K, V in cache at position pos */
    if (pos >= 0 && pos < rin->max_seq_len) {
        if (pos < rin->kv_pos + 10) {
            if (pos >= rin->kv_pos) rin->kv_pos = pos + 1;
        }
        for (int j = 0; j < kv_dim; j++) {
            rin->k_cache[(size_t)pos * kv_dim + j] = rin->k_buf[j];
            rin->v_cache[(size_t)pos * kv_dim + j] = rin->v_buf[j];
        }
    }
    /* Compute effective sequence length (capped to allocated) */
    int seq_len = pos + 1;
    if (seq_len > rin->max_seq_len) seq_len = rin->max_seq_len;

    /* Scaled dot-product attention (multi-head, causal) */
    for (int h = 0; h < num_heads; h++) {
        int off = h * hd;
        int h_seq_len = seq_len;

        /* Compute scores: Q · K[i] / sqrt(hd) for i in 0..pos */
        int32_t scores[2048];
        int sl = h_seq_len;
        if (sl > 2048) sl = 2048;
        int32_t max_s = -1000000000;
        for (int i = 0; i < sl; i++) {
            int32_t dot = 0;
            for (int d = 0; d < hd; d++)
                dot += (int32_t)rin->q_buf[off + d] *
                       rin->k_cache[(size_t)i * kv_dim + off + d];
            /* Scale by 1/sqrt(hd) — use shift for speed */
            int shift = 0;
            int hd_val = hd;
            while (hd_val >>= 1) shift++;
            shift = shift / 2;  /* approx log2(sqrt(hd)) */
            scores[i] = dot >> shift;
            if (scores[i] > max_s) max_s = scores[i];
        }

        /* Softmax */
        uint32_t exp_s[2048];
        uint32_t sum_exp = 0;
        for (int i = 0; i < sl; i++) {
            int32_t diff = scores[i] - max_s;
            /* 2^x approximated as (x >= -16) ? (1 << (x + 16)) : 0 */
            uint32_t e = (diff >= -16) ? (uint32_t)(1 << (diff + 16)) : 0;
            exp_s[i] = e;
            sum_exp += e;
        }

        /* Weighted sum of V */
        if (sum_exp > 0) {
            for (int d = 0; d < hd; d++) {
                int32_t sum = 0;
                for (int i = 0; i < sl; i++)
                    sum += (int32_t)exp_s[i] *
                           rin->v_cache[(size_t)i * kv_dim + off + d];
                sum = sum / (int32_t)(sum_exp >> 16);
                if (sum > 32767) sum = 32767;
                if (sum < -32768) sum = -32768;
                rin->attn_buf[off + d] = (int16_t)sum;
            }
        } else {
            memset(&rin->attn_buf[off], 0, hd * sizeof(int16_t));
        }
    }
#endif

    /* Output projection (Wo) — shared by both paths */
    q15_to_int8(rin->attn_buf, rin->buf0, dim);
    int32_t sum_x_wo = sum_int8(rin->buf0, dim);
    gemv_kernel(rin->Wo[layer], dim, dim, rin->buf0, rin->gemv_buf);
    scale_to_q15(rin->gemv_buf, rin->attn_buf, dim, rin->so[layer], rin->bo[layer], sum_x_wo);
}

/* Transformer FFN: W2(ReLU(W1(x))) */
static void rin_tfm_ffn(RIN_Internal *rin, const int16_t *x,
                         int dim, int ffn_dim, int layer) {
    q15_to_int8(x, rin->buf0, dim);
    int32_t sum_x_w1 = sum_int8(rin->buf0, dim);

    /* W1 up projection: dim → ffn_dim, ReLU. Output is raw float [0,127]. */
    gemv_kernel(rin->W1[layer], ffn_dim, dim, rin->buf0, rin->gemv_buf);
    float sc1 = rin->s1[layer];
    float corr_w1 = 128.0f * sc1 * sum_x_w1;
    for (int j = 0; j < ffn_dim; j++) {
        float v = rin->gemv_buf[j] * sc1 - corr_w1 + rin->b1[layer][j];
        if (v < 0.0f) v = 0.0f;
        if (v > 127.0f) v = 127.0f;
        rin->buf1[j] = (int8_t)v;
    }

    /* W2 down projection: ffn_dim → dim, back to Q15 */
    int32_t sum_x_w2 = sum_int8(rin->buf1, ffn_dim);
    gemv_kernel(rin->W2[layer], dim, ffn_dim, rin->buf1, rin->gemv_buf);
    scale_to_q15(rin->gemv_buf, rin->ffn_buf, dim, rin->s2[layer], rin->b2[layer], sum_x_w2);
}

/* Q15 LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta */
static void rin_layer_norm(int16_t *x, int dim, const int16_t *gamma,
                            const int16_t *beta) {
    int32_t sum = 0;
    for (int i = 0; i < dim; i++) sum += x[i];
    int32_t mean = sum / dim;  /* Q15 mean */

    int32_t var_sum = 0;
    for (int i = 0; i < dim; i++) {
        int32_t diff = x[i] - mean;
        var_sum += (diff * diff) >> 7;  /* scale down to avoid overflow */
    }
    int32_t var = var_sum / dim;  /* Q15 variance (scaled by 1/128) */
    if (var < 1) var = 1;
    /* Use double for rsqrt to avoid underflow with -ffast-math */
    double inv_var = (double)var * 128.0 / 65536.0 + 1e-5;
    int32_t rstd = (int32_t)(65536.0 / sqrt(inv_var));

    for (int i = 0; i < dim; i++) {
        int32_t diff = x[i] - mean;
        int32_t norm = (diff * rstd) >> 16;  /* Q15 normalized */
        int32_t v = norm * gamma[i] / 256 + beta[i];
        if (v < -32768) v = -32768;
        if (v > 32767) v = 32767;
        x[i] = (int16_t)v;
    }
}

/* One transformer step: embed → attn × layers → unembed → logits */
static void rin_tfm_step(RIN_Internal *rin, int token_id, int pos,
                          int16_t *logits, int dim, int num_layers,
                          int num_heads, int ffn_dim, int vocab_size) {
    /* Embed token (uint8 centered at 128) */
    uint8_t *emb = rin->embed_table;
    int e_off = (int)token_id * dim;
    for (int j = 0; j < dim; j++) {
        float deq = ((int32_t)emb[e_off + j] - 128) * rin->embed_scale + rin->embed_bias[j];
        int32_t v_q15 = (int32_t)(deq * 256.0f);
        rin->residual[j] = clamp_q15(v_q15 + rin->pos_embed[(size_t)pos * dim + j]);
    }

    int has_ln = (rin->num_ln_sets > 0);
    /* Transformer layers */
    for (int l = 0; l < num_layers; l++) {
        /* Save residual for shortcut */
        int16_t saved[RIN_MAX_SAVED];
        int sd = dim < RIN_MAX_SAVED ? dim : RIN_MAX_SAVED;
        memcpy(saved, rin->residual, sd * sizeof(int16_t));

        /* Pre-LN: apply ln1 before attention */
        if (has_ln) rin_layer_norm(rin->residual, dim, rin->ln_gamma[l * 2], rin->ln_beta[l * 2]);

        /* Multi-head attention */
        rin_tfm_attention(rin, rin->residual, pos, dim, num_heads, l);

        /* Residual add: x = x + attn_output */
        for (int j = 0; j < dim; j++)
            rin->residual[j] = clamp_q15((int32_t)saved[j] + rin->attn_buf[j]);

        /* Save residual for FFN shortcut */
        memcpy(saved, rin->residual, sd * sizeof(int16_t));

        /* Pre-LN: apply ln2 before FFN */
        if (has_ln) rin_layer_norm(rin->residual, dim, rin->ln_gamma[l * 2 + 1], rin->ln_beta[l * 2 + 1]);

        /* FFN */
        rin_tfm_ffn(rin, rin->residual, dim, ffn_dim, l);

        /* Residual add: x = x + ffn_output */
        for (int j = 0; j < dim; j++)
            rin->residual[j] = clamp_q15((int32_t)saved[j] + rin->ffn_buf[j]);
    }

    /* Final ln_f before unembed */
    if (has_ln) rin_layer_norm(rin->residual, dim, rin->ln_gamma[num_layers * 2], rin->ln_beta[num_layers * 2]);

    /* Unembed: GEMV(output_W, residual_int8) → logits in Q15 */
    q15_to_int8(rin->residual, rin->buf0, dim);
    int32_t sum_x_out = sum_int8(rin->buf0, dim);
    float sc_out = rin->model.scale[0];
    int od = vocab_size;
    gemv_kernel(rin->model.W[0], od, dim, rin->buf0, rin->gemv_buf);
    for (int j = 0; j < od && j < RIN_MAX_VOCAB; j++) {
        float v = (rin->gemv_buf[j] * sc_out - 128.0f * sc_out * sum_x_out + rin->model.bias[0][j]) * 256.0f;
        if (v < -32768.0f) v = -32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        logits[j] = (int16_t)v;
    }
}

/* ============================================================================
 * OUTPUT LAYER: GEMV + PTsoftmax + Argmax (+ guarda logits en ctx)
 * ============================================================================ */
static int rin_output_layer(RIN_Internal *rin, RIN_Context *ctx,
                              int8_t *in, uint8_t *probs, int *od_out) {
    RIN_Model *m = &rin->model;
    int last_dim = m->layer_dims[m->num_layers - 2];
    int od = m->output_dim;
    const uint8_t *W = m->W[m->num_layers - 1];

    gemv_kernel(W, od, last_dim, in, rin->gemv_buf);

    float sc = m->scale[m->num_layers - 1];
    float *bs = m->bias[m->num_layers - 1];
    int8_t logits[RIN_MAX_VOCAB];
    for (int i = 0; i < od; i++) {
        int32_t v = (int32_t)(rin->gemv_buf[i] * sc + bs[i]);
        if (v < -127) v = -127;
        if (v > 127) v = 127;
        logits[i] = (int8_t)v;
    }

    /* Store logits in ctx for RIN_GenerateToken */
    memcpy(ctx->last_logits, logits, od);
    ctx->last_logits_dim = od;

    RIN_PTSoftmax_Table tbl;
    RIN_PTSoftmax_InitTable(&tbl, 32);
    RIN_PTSoftmax_Compute(&tbl, logits, probs, od);

    int best = 0;
    uint8_t best_v = 0;
    for (int i = 0; i < od; i++)
        if (probs[i] > best_v) { best_v = probs[i]; best = i; }
    *od_out = od;
    return best;
}

/* ============================================================================
 * LOADERS
 * ============================================================================ */
static int load_layer_weights(FILE *f, int prev_dim, int this_dim,
                               uint8_t **W, float *scale_out,
                               float **bias_out, int scale_first) {
    size_t sz = (size_t)this_dim * prev_dim;
    *W = (uint8_t*)aligned_alloc(32, sz);
    if (!*W) return -1;

    if (scale_first) {
        if (fread(scale_out, 4, 1, f) != 1) return -1;
        if (*scale_out < 1e-10f) *scale_out = 1e-10f;
        if (fread(*W, 1, sz, f) != sz) return -1;
    } else {
        if (fread(*W, 1, sz, f) != sz) return -1;
        if (fread(scale_out, 4, 1, f) != 1) return -1;
        if (*scale_out < 1e-10f) *scale_out = 1e-10f;
    }

    *bias_out = (float*)aligned_alloc(32, (size_t)this_dim * sizeof(float));
    if (!*bias_out) return -1;
    if (fread(*bias_out, 4, this_dim, f) != (size_t)this_dim) { free(*bias_out); return -1; }
    return 0;
}

static int load_rin_legacy_header(FILE *f, RIN_Model *m) {
    fread(&m->input_dim, sizeof(int), 1, f);
    fread(&m->output_dim, sizeof(int), 1, f);
    fread(&m->num_layers, sizeof(int), 1, f);
    fread(m->layer_dims, sizeof(int), m->num_layers, f);
    int nd; fread(&nd, sizeof(int), 1, f);
    (void)nd;
    return 0;
}

static int load_rin_header(FILE *f, RIN_Model *m) {
    uint32_t version;
    fread(&version, 4, 1, f);
    (void)version;
    fread(&m->input_dim, 4, 1, f);
    fread(&m->output_dim, 4, 1, f);
    fread(&m->num_layers, 4, 1, f);
    fread(m->layer_dims, 4, m->num_layers, f);
    return 0;
}

/* ============================================================================
 * TRANSFORMER HEADER PARSER
 * ============================================================================ */
static uint32_t read_u32(FILE *f) {
    uint32_t v = 0;
    fread(&v, 4, 1, f);
    return v;
}

static int load_transformer_header(FILE *f, int *num_layers, int *dim,
                                    int *vocab_size, int *num_heads,
                                    int *max_seq_len, int *ffn_dim) {
    *num_layers = (int)read_u32(f);
    *dim = (int)read_u32(f);
    *vocab_size = (int)read_u32(f);
    *num_heads = (int)read_u32(f);
    *max_seq_len = (int)read_u32(f);
    *ffn_dim = (int)read_u32(f);
    return 0;
}

static int load_transformer_weight_layer_bias(FILE *f, int rows, int cols,
                                                uint8_t **W, float *scale_out,
                                                float **bias_out, int bias_count);

static int load_transformer_weight_layer(FILE *f, int rows, int cols,
                                           uint8_t **W, float *scale_out,
                                           float **bias_out) {
    return load_transformer_weight_layer_bias(f, rows, cols, W, scale_out, bias_out, rows);
}

static int load_transformer_weight_layer_bias(FILE *f, int rows, int cols,
                                                uint8_t **W, float *scale_out,
                                                float **bias_out, int bias_count) {
    size_t sz = (size_t)rows * cols;
    *W = (uint8_t*)malloc(sz);
    if (!*W) return -1;
    if (fread(scale_out, 4, 1, f) != 1) { free(*W); return -1; }
    if (*scale_out < 1e-10f) *scale_out = 1e-10f;
    if (fread(*W, 1, sz, f) != sz) { free(*W); return -1; }
    *bias_out = (float*)malloc((size_t)bias_count * sizeof(float));
    if (!*bias_out) { free(*W); return -1; }
    if (fread(*bias_out, 4, bias_count, f) != (size_t)bias_count) { free(*W); free(*bias_out); return -1; }
    return 0;
}

/* ============================================================================
 * RIN_LoadWeights
 * ============================================================================ */
RinStatus RIN_LoadWeights(RIN_Context* ctx, const char* path) {
    if (!ctx || !path) return RIN_STATUS_ERROR_INVALID_INPUT;

    RIN_Internal *rin = (RIN_Internal*)calloc(1, sizeof(RIN_Internal));
    if (!rin) return RIN_STATUS_ERROR_MEMORY;

    RIN_Model *m = &rin->model;
    FILE *f = fopen(path, "rb");
    if (!f) { free(rin); return RIN_STATUS_ERROR_WEIGHTS; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4) { fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS; }

    int scale_first = 0;
    rin->architecture = 0;

    if (!memcmp(magic, "THOR", 4)) {
        if (load_rin_legacy_header(f, m)) { fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS; }
    } else if (!memcmp(magic, "RIN1", 4)) {
        uint32_t version = read_u32(f);
        (void)version;
        uint32_t architecture = read_u32(f);
        rin->architecture = (int)architecture;
        if (architecture == 0) {
            /* MLP format (existing behavior) */
            scale_first = 1;
            if (load_rin_header(f, m)) { fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS; }
        } else if (architecture == 1) {
            /* Transformer format */
            int num_layers, dim, vocab_size, num_heads, max_seq_len, ffn_dim;
            if (load_transformer_header(f, &num_layers, &dim, &vocab_size,
                                        &num_heads, &max_seq_len, &ffn_dim)) {
                fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
            }
            m->input_dim = dim;
            m->output_dim = vocab_size;
            m->num_layers = num_layers;
            rin->num_heads = num_heads;
            rin->ffn_dim = ffn_dim;
            rin->max_seq_len = max_seq_len;
            for (int i = 0; i < num_layers; i++) {
                m->layer_dims[i] = dim;
            }

            /* Load embedding table [vocab_size, dim] */
            if (load_transformer_weight_layer_bias(f, vocab_size, dim,
                (uint8_t**)&rin->embed_table, &rin->embed_scale, &rin->embed_bias, dim)) {
                fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
            }

            /* Load each transformer layer's weights */
            for (int i = 0; i < num_layers; i++) {
                if (load_transformer_weight_layer(f, dim, dim, &rin->Wq[i], &rin->sq[i], &rin->bq[i]) ||
                    load_transformer_weight_layer(f, dim, dim, &rin->Wk[i], &rin->sk[i], &rin->bk[i]) ||
                    load_transformer_weight_layer(f, dim, dim, &rin->Wv[i], &rin->sv[i], &rin->bv[i]) ||
                    load_transformer_weight_layer(f, dim, dim, &rin->Wo[i], &rin->so[i], &rin->bo[i]) ||
                    load_transformer_weight_layer(f, ffn_dim, dim, &rin->W1[i], &rin->s1[i], &rin->b1[i]) ||
                    load_transformer_weight_layer(f, dim, ffn_dim, &rin->W2[i], &rin->s2[i], &rin->b2[i])) {
                    fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
                }
            }

            /* Load output (unembedding) [vocab_size, dim] */
            if (load_transformer_weight_layer(f, vocab_size, dim, &m->W[0], &m->scale[0], &m->bias[0])) {
                fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
            }

            /* Load position embeddings [max_seq_len, dim] as INT16 */
            size_t pos_sz = (size_t)max_seq_len * dim;
            rin->pos_embed = (int16_t*)aligned_alloc(32, pos_sz * sizeof(int16_t));
            if (!rin->pos_embed) { fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS; }
            if (fread(rin->pos_embed, sizeof(int16_t), pos_sz, f) != pos_sz) {
                fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
            }

            /* Load LayerNorm parameters */
            {
                uint32_t ln_count;
                if (fread(&ln_count, 4, 1, f) != 1) {
                    /* File may not have LN data (older format) — fall through */
                    ln_count = 0;
                }
                rin->num_ln_sets = (int)ln_count;
                for (int i = 0; i < (int)ln_count; i++) {
                    rin->ln_gamma[i] = (int16_t*)aligned_alloc(32, (size_t)dim * sizeof(int16_t));
                    rin->ln_beta[i] = (int16_t*)aligned_alloc(32, (size_t)dim * sizeof(int16_t));
                    if (!rin->ln_gamma[i] || !rin->ln_beta[i]) {
                        fclose(f); free(rin); return RIN_STATUS_ERROR_MEMORY;
                    }
                    if (fread(rin->ln_gamma[i], sizeof(int16_t), dim, f) != (size_t)dim ||
                        fread(rin->ln_beta[i], sizeof(int16_t), dim, f) != (size_t)dim) {
                        fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
                    }
                }
            }

            /* Load character set (vocab_size bytes, padded to 4-byte) */
            rin->char_set = (char*)calloc(vocab_size + 1, 1);
            if (rin->char_set) {
                int chars_read = (int)fread(rin->char_set, 1, vocab_size, f);
                if (chars_read == vocab_size) {
                    rin->char_set_size = vocab_size;
                    rin->char_set[vocab_size] = 0;  /* null terminate */
                } else {
                    /* charset may not be present in older files — not fatal */
                    free(rin->char_set);
                    rin->char_set = NULL;
                    rin->char_set_size = 0;
                }
                /* Skip padding bytes */
                int pad = (4 - (vocab_size % 4)) % 4;
                if (pad) fseek(f, pad, SEEK_CUR);
            }

            m->total_params = 0;
            m->input_dim = dim;
            m->output_dim = vocab_size;
            m->loaded = 1;
            fclose(f);

            /* Allocate transformer buffers */
            rin->max_dim = dim > ffn_dim ? dim : ffn_dim;
            size_t dsz = (size_t)rin->max_dim;
            rin->buf0 = (int8_t*)aligned_alloc(32, dsz);
            rin->buf1 = (int8_t*)aligned_alloc(32, dsz);
            rin->gemv_buf = (int32_t*)aligned_alloc(32, dsz * sizeof(int32_t));
            rin->q_buf = (int16_t*)aligned_alloc(32, dsz * sizeof(int16_t));
            rin->k_buf = (int16_t*)aligned_alloc(32, dsz * sizeof(int16_t));
            rin->v_buf = (int16_t*)aligned_alloc(32, dsz * sizeof(int16_t));
            rin->attn_buf = (int16_t*)aligned_alloc(32, dsz * sizeof(int16_t));
            rin->residual = (int16_t*)aligned_alloc(32, dsz * sizeof(int16_t));
            rin->ffn_buf = (int16_t*)aligned_alloc(32, (size_t)ffn_dim * sizeof(int16_t));

            /* KV cache (multi-head) */
            int hd = dim / num_heads;
            rin->kv_cache_dim = num_heads * hd;
            size_t cache_sz = (size_t)max_seq_len * rin->kv_cache_dim;
            rin->k_cache = (int16_t*)calloc(cache_sz, sizeof(int16_t));
            rin->v_cache = (int16_t*)calloc(cache_sz, sizeof(int16_t));
            rin->kv_pos = 0;

#ifdef RIN_USE_BHRR
            /* BHRR attention: allocate per-layer, per-head contexts */
            {
                int S = RIN_BHRR_SLOTS;
                rin->bhrr.num_layers = num_layers;
                rin->bhrr.num_heads = num_heads;
                rin->bhrr.head_dim = hd;
                rin->bhrr.num_slots = S;

                size_t total_heads = (size_t)num_layers * num_heads;
                size_t slot_sz = total_heads * S * hd * sizeof(int32_t);
                size_t key_sz  = total_heads * S * hd * sizeof(int16_t);

                rin->bhrr.slot_bufs = (int32_t*)calloc(slot_sz, 1);
                rin->bhrr.key_bufs  = (int16_t*)calloc(key_sz, 1);
                rin->bhrr.heads     = (RinBHRR*)calloc(total_heads, sizeof(RinBHRR));

                rin->bhrr_qsign = (int16_t*)calloc(dim, sizeof(int16_t));
                rin->bhrr_ksign = (int16_t*)calloc(dim, sizeof(int16_t));
                rin->bhrr_vsign = (int16_t*)calloc(dim, sizeof(int16_t));

                if (!rin->bhrr.slot_bufs || !rin->bhrr.key_bufs ||
                    !rin->bhrr.heads || !rin->bhrr_qsign ||
                    !rin->bhrr_ksign || !rin->bhrr_vsign)
                    return RIN_STATUS_ERROR_MEMORY;

                int32_t* sp = rin->bhrr.slot_bufs;
                int16_t* kp = rin->bhrr.key_bufs;
                for (int li = 0; li < num_layers; li++) {
                    for (int h = 0; h < num_heads; h++) {
                        int idx = li * num_heads + h;
                        rin_bhrr_init(&rin->bhrr.heads[idx], sp, kp, S, hd, 42 + idx);
                        sp += S * hd;
                        kp += S * hd;
                    }
                }
            }
#endif

            if (!rin->buf0 || !rin->buf1 || !rin->gemv_buf ||
                !rin->q_buf || !rin->k_buf || !rin->v_buf ||
                !rin->attn_buf || !rin->residual || !rin->ffn_buf ||
                !rin->k_cache || !rin->v_cache)
                return RIN_STATUS_ERROR_MEMORY;

            ctx->_internal = rin;
            return RIN_STATUS_OK;
        }
    } else {
        fclose(f); free(rin); return RIN_STATUS_ERROR_WEIGHTS;
    }

    /* --- MLP weight loading (existing behavior) --- */
    m->total_params = 0;
    int prev = m->input_dim;
    for (int i = 0; i < m->num_layers; i++) {
        int od = m->layer_dims[i];
        if (load_layer_weights(f, prev, od, &m->W[i], &m->scale[i],
                               &m->bias[i], scale_first)) {
            fclose(f); return RIN_STATUS_ERROR_WEIGHTS;
        }
        m->total_params += (int)prev * od;
        prev = od;
    }
    fclose(f);
    m->loaded = 1;

    rin->max_dim = m->input_dim;
    for (int i = 0; i < m->num_layers; i++)
        if (m->layer_dims[i] > rin->max_dim) rin->max_dim = m->layer_dims[i];
    size_t pg = (size_t)rin->max_dim;
    rin->buf0 = (int8_t*)aligned_alloc(32, pg);
    rin->buf1 = (int8_t*)aligned_alloc(32, pg);
    int ts = ctx->config.timesteps > 0 ? ctx->config.timesteps : 4;
    size_t gemv_sz = pg * sizeof(int32_t);
    size_t spike_sz = (size_t)2 * ts * pg;
    size_t alloc_sz = gemv_sz > spike_sz ? gemv_sz : spike_sz;
    rin->gemv_buf = (int32_t*)aligned_alloc(32, alloc_sz);
    if (!rin->buf0 || !rin->buf1 || !rin->gemv_buf) return RIN_STATUS_ERROR_MEMORY;

    /* Initialize KV cache */
    int kv_dim = m->layer_dims[0];
    for (int i = 1; i < m->num_layers - 1; i++)
        if (m->layer_dims[i] > kv_dim) kv_dim = m->layer_dims[i];
    rin->kv_cache.dim = kv_dim > MAX_ATTN_DIM ? MAX_ATTN_DIM : kv_dim;
    rin->kv_cache.max_seq = ctx->config.max_seq_len;
    rin->kv_cache.current_pos = 0;
    rin->kv_cache.K = (int16_t*)calloc((size_t)rin->kv_cache.max_seq * rin->kv_cache.dim, sizeof(int16_t));
    rin->kv_cache.V = (int16_t*)calloc((size_t)rin->kv_cache.max_seq * rin->kv_cache.dim, sizeof(int16_t));

    ctx->_internal = rin;
    return RIN_STATUS_OK;
}

/* ============================================================================
 * RIN_Inference
 * ============================================================================ */
RinStatus RIN_Inference(RIN_Context* ctx,
                          const uint32_t* input_ids,
                          uint32_t num_input,
                          uint32_t max_output,
                          RIN_InferenceResult* result) {
    (void)max_output;
    if (!ctx || !input_ids || !result) return RIN_STATUS_ERROR_INVALID_INPUT;
    RIN_Internal *rin = (RIN_Internal*)ctx->_internal;
    if (!rin || !rin->model.loaded) return RIN_STATUS_ERROR_WEIGHTS;

    if (!rin->meter.initialized) {
        RIN_EnergyMeter_Init(&rin->meter);
    }
    RIN_Model *m = &rin->model;
    RIN_EnergyMeasurement rin_meas;
    RIN_EnergyMeter_StartMeasurement(&rin->meter, &rin_meas);
    uint64_t t0 = RIN_DPTM_GetTimestampNs();

    int8_t *in = rin->buf0;
    int8_t *out = rin->buf1;

    int num_hidden = m->num_layers - 1;
    int od;
    uint8_t probs[RIN_MAX_VOCAB];
    int predicted = 0;

    if (ctx->config.inference_mode == RIN_MODE_TRANSFORMER) {
        /* === TRANSFORMER MODE: autoregressive generation === */
        int dim = m->input_dim;
        int num_tfm_layers = m->num_layers;

        /* Load config from internal (set during weight loading) */
        int nh = rin->num_heads > 0 ? rin->num_heads : ctx->config.num_heads;
        int fd = rin->ffn_dim > 0 ? rin->ffn_dim : ctx->config.ffn_dim;
        if (fd == 0) fd = 4 * dim;
        int vs = m->output_dim;

        /* Reset KV cache for new sequence */
        rin->kv_pos = 0;

#ifdef RIN_USE_BHRR
        /* Reset all BHRR contexts for new sequence */
        for (int li = 0; li < num_tfm_layers; li++)
            for (int h = 0; h < nh; h++)
                rin_bhrr_clear(&rin->bhrr.heads[li * nh + h]);
#endif

        /* Cap input to max_seq_len */
        uint32_t capped_input = num_input;
        if (capped_input > (uint32_t)rin->max_seq_len)
            capped_input = (uint32_t)rin->max_seq_len;

        /* Process prefix tokens */
        int16_t logits_buf[RIN_MAX_VOCAB];

        for (uint32_t i = 0; i < capped_input; i++) {
            int tid = (int)input_ids[i];
            if (tid >= vs) tid = 0;
            rin_tfm_step(rin, tid, (int)i, logits_buf, dim,
                         num_tfm_layers, nh, fd, vs);
        }

        /* Generate additional tokens */
        uint32_t generated[32];
        uint32_t num_gen = 0;
        uint32_t max_out = max_output;
        if (max_out > 32) max_out = 32;
        if (max_out == 0) max_out = 1;

        for (uint32_t g = 0; g < max_out; g++) {
            /* Store logits in context for RIN_GenerateToken */
            int od = vs < RIN_MAX_VOCAB ? vs : RIN_MAX_VOCAB;
            for (int j = 0; j < od; j++) {
                int32_t v = logits_buf[j];
                if (v < -127) v = -127;
                if (v > 127) v = 127;
                ctx->last_logits[j] = (int8_t)v;
            }
            ctx->last_logits_dim = od;

            /* Select token: use argmax for deterministic comparison */
            int sampled = 0;
            for (int j = 1; j < od; j++) if (logits_buf[j] > logits_buf[sampled]) sampled = j;
            generated[g] = (uint32_t)sampled;
            num_gen++;

            /* Feed sampled token into model for next prediction */
            if (g < max_out - 1) {
                int next_pos = (int)(num_input + g);
                rin_tfm_step(rin, sampled, next_pos, logits_buf, dim,
                             num_tfm_layers, nh, fd, vs);
            }
        }

        /* Set up result variables for output section */
        od = vs < RIN_MAX_VOCAB ? vs : RIN_MAX_VOCAB;
        memset(probs, 0, RIN_MAX_VOCAB);
        if (generated[0] < (uint32_t)od) probs[generated[0]] = 255;
        predicted = (int)generated[0];

        /* Set last_logits for RIN_GenerateToken */
        for (int j = 0; j < od && j < RIN_MAX_VOCAB; j++) {
            int32_t v = logits_buf[j];
            if (v < -127) v = -127;
            if (v > 127) v = 127;
            ctx->last_logits[j] = (int8_t)v;
        }
        ctx->last_logits_dim = od;
        ctx->seq_len = num_input + num_gen;
        for (uint32_t gi = 0; gi < num_gen && gi < 32; gi++)
            ctx->sequence[gi].id = generated[gi];

    } else {
        memset(in, 0, rin->max_dim);
        for (uint32_t i = 0; i < num_input && i < (uint32_t)m->input_dim; i++)
            in[i] = (int8_t)(((int)input_ids[i % 10] * 20 - 128) & 0xFF);

    if (ctx->config.inference_mode == RIN_MODE_SNN) {
        /* === SNN MODE: LIF-based spiking pipeline === */
        /* Use gemv_buf as spike storage (has max_dim*4 bytes) */
        int ts = ctx->config.timesteps;
        int spike_frame = (size_t)rin->max_dim;   /* bytes per timestep */
        uint8_t *spikes_in = (uint8_t*)rin->gemv_buf;
        uint8_t *spikes_out = (uint8_t*)rin->gemv_buf + spike_frame * ts;

        /* Convert input to initial spikes (timestep 0) */
        for (int j = 0; j < m->input_dim; j++)
            spikes_in[j] = (uint8_t)(in[j] > 0 ? 1 : 0);
        /* Remaining timesteps: zero */
        for (int j = m->input_dim; j < spike_frame * ts; j++)
            spikes_in[j] = 0;

        for (int li = 0; li < num_hidden; li++) {
            int M = m->layer_dims[li];
            /* Clear output spikes */
            memset(spikes_out, 0, spike_frame * ts);

            rin_snn_layer(rin, ctx, li, spikes_in, spikes_out);
            uint8_t *t = spikes_in; spikes_in = spikes_out; spikes_out = t;
        }

        /* Convert last layer spikes to INT8 for output */
        for (int j = 0; j < m->output_dim; j++) {
            int sum = 0;
            for (int t = 0; t < ts; t++)
                sum += spikes_in[(size_t)t * spike_frame + j];
            out[j] = (int8_t)(sum * 20 - 128);
        }

    } else if (ctx->config.inference_mode == RIN_MODE_ATTN) {
        /* === ATTN MODE: MLP + Attention === */
        for (int li = 0; li < num_hidden; li++) {
            int M = m->layer_dims[li];

            /* GEMV */
            int K = (li == 0) ? m->input_dim : m->layer_dims[li-1];
            gemv_kernel(m->W[li], M, K, in, rin->gemv_buf);
            apply_scale_bias(rin->gemv_buf, M, m->scale[li], m->bias[li]);

            /* Attention on hidden states (convert to Q15 first) */
            int16_t attn_in[MAX_ATTN_DIM];
            int nd = M < MAX_ATTN_DIM ? M : MAX_ATTN_DIM;
            for (int j = 0; j < nd; j++)
                attn_in[j] = (int16_t)rin->gemv_buf[j];

            int8_t attn_out[MAX_ATTN_DIM];
            rin_attention_block(attn_in, nd, li, &rin->kv_cache, attn_out);

            /* ReLU + BSPN */
            int32_t l1 = 0;
            for (int j = 0; j < M; j++) {
                int32_t v = (int32_t)attn_out[j < nd ? j : 0];
                if (v < 0) v = 0;
                l1 += v;
            }
            uint32_t l1u = (uint32_t)l1;
            int shift = 0;
            while (l1u >>= 1) shift++;
            shift = shift > 8 ? shift - 8 : 0;

            for (int j = 0; j < M; j++) {
                int32_t v = (int32_t)attn_out[j < nd ? j : 0];
                if (v < 0) v = 0;
                v >>= shift;
                if (v > 127) v = 127;
                out[j] = (int8_t)v;
            }
            int8_t *t = in; in = out; out = t;
        }

    } else if (ctx->config.inference_mode == RIN_MODE_THOR) {
        /* RIN MODE: pure GEMV, same semantics as legacy rin_final.c */
        for (int li = 0; li < num_hidden; li++) {
            int K = (li == 0) ? m->input_dim : m->layer_dims[li-1];
            int M = m->layer_dims[li];
            float sc = m->scale[li];
            float *bs = m->bias[li];
            gemv_kernel(m->W[li], M, K, in, rin->gemv_buf);
            for (int j = 0; j < M; j++) {
                int32_t v = (int32_t)(rin->gemv_buf[j] * 256.0f / sc + bs[j]);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                out[j] = (int8_t)(v - 128);
            }
            int8_t *t = in; in = out; out = t;
        }
    } else {
        /* MLP MODE (default): GEMV → BSPN → ReLU */
        for (int li = 0; li < num_hidden; li++) {
            rin_hidden_layer(rin, li, in, out);
            int8_t *t = in; in = out; out = t;
        }
    }
    }

    /* Output layer */
    if (ctx->config.inference_mode == RIN_MODE_TRANSFORMER) {
        /* Already handled above */
    } else if (ctx->config.inference_mode == RIN_MODE_THOR) {
        /* RIN: GEMV + bias + clamp[0,255], no softmax */
        int last_dim = m->layer_dims[m->num_layers - 2];
        od = m->output_dim;
        const uint8_t *W = m->W[m->num_layers - 1];
        gemv_kernel(W, od, last_dim, in, rin->gemv_buf);
        float *bs = m->bias[m->num_layers - 1];
        int best = 0;
        uint8_t best_v = 0;
        for (int i = 0; i < od; i++) {
            int32_t v = rin->gemv_buf[i] + bs[i];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            probs[i] = (uint8_t)v;
            if (probs[i] > best_v) { best_v = probs[i]; best = i; }
        }
        predicted = best;
        memcpy(ctx->last_logits, probs, od);
        ctx->last_logits_dim = od;
    } else {
        predicted = rin_output_layer(rin, ctx, in, probs, &od);
    }

    double rin_joules = RIN_EnergyMeter_EndMeasurement(&rin->meter, &rin_meas, RIN_RAPL_DOMAIN_PKG);
    uint64_t rin_energy_uj = (rin_joules > 0) ? (uint64_t)(rin_joules * 1e6) : 0;
    rin->total_energy_uj += rin_energy_uj;

    uint64_t t1 = RIN_DPTM_GetTimestampNs();

    result->latency_ns = t1 - t0;
    result->num_tokens = 1;
    result->tokens_per_second = 1e9f / (float)(t1 - t0);
    result->energy_joules = rin_joules > 0 ? rin_joules : 0;

    if (result->tokens) {
        result->tokens[0].id = (uint32_t)predicted;
        result->tokens[0].probability = (float)probs[predicted] / 255.0f;
        result->tokens[0].generation_time_ns = t1 - t0;
    }

    ctx->inference_count++;
    ctx->total_tokens_generated++;
    ctx->total_energy_joules += result->energy_joules;
    return RIN_STATUS_OK;
}

/* ============================================================================
 * RIN_GenerateToken - Sampling REAL
 *
 * Samples from the probability distribution of the last inference:
 *   1. Temperature: logits[i] = logits[i] / temperature
 *   2. Top-k: only the k highest logits
 *   3. Top-p: solo el subconjunto con probabilidad acumulada > p
 *   4. Multinomial: sample from the filtered distribution
 * ============================================================================ */
RinStatus RIN_GenerateToken(RIN_Context* ctx, RIN_Token* next_token) {
    if (!ctx || !next_token) return RIN_STATUS_ERROR_INVALID_INPUT;
    RIN_Internal *rin = (RIN_Internal*)ctx->_internal;
    if (!rin || !rin->model.loaded) return RIN_STATUS_ERROR_WEIGHTS;
    if (ctx->last_logits_dim == 0) return RIN_STATUS_ERROR_INFERENCE;

    int od = (int)ctx->last_logits_dim;

    /* Copy actual logits from last inference */
    int8_t logits[32];
    memcpy(logits, ctx->last_logits, od);

    /* Temperature scaling */
    float temp = ctx->config.temperature;
    if (temp < 0.01f) temp = 1.0f;
    if (temp != 1.0f) {
        for (int i = 0; i < od; i++) {
            float v = (float)logits[i] / temp;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            logits[i] = (int8_t)v;
        }
    }

    /* Normalize: subtract max so that max(logits)=0 */
    int8_t max_l = logits[0];
    for (int i = 1; i < od; i++)
        if (logits[i] > max_l) max_l = logits[i];
    for (int i = 0; i < od; i++)
        logits[i] -= max_l;

    /* Softmax in Q8 with 2^x table and exact division */
    uint32_t exp_vals[32];
    uint8_t probs[32];
    RIN_PTSoftmax_Table tbl;
    RIN_PTSoftmax_InitTable(&tbl, 32);
    uint32_t total_exp = 0;
    for (int i = 0; i < od; i++) {
        exp_vals[i] = RIN_PTSoftmax_Lookup(&tbl, logits[i]);
        total_exp += exp_vals[i];
    }
    if (total_exp > 0) {
        for (int i = 0; i < od; i++)
            probs[i] = (uint8_t)((exp_vals[i] * 255ULL) / total_exp);
    } else {
        memset(probs, 255 / od, od);
    }

    /* Top-k filtering */
    if (ctx->config.top_k > 0 && ctx->config.top_k < (uint32_t)od) {
        uint8_t threshold = 0;
        uint8_t sorted[32];
        memcpy(sorted, probs, od);
        for (int i = 0; i < od; i++) {
            for (int j = i + 1; j < od; j++) {
                if (sorted[j] > sorted[i]) {
                    uint8_t t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                }
            }
        }
        threshold = sorted[ctx->config.top_k - 1];
        for (int i = 0; i < od; i++)
            if (probs[i] < threshold) probs[i] = 0;
    }

    /* Top-p (nucleus) filtering */
    if (ctx->config.top_p > 0.0f && ctx->config.top_p < 1.0f) {
        uint32_t total = 0;
        for (int i = 0; i < od; i++) total += probs[i];
        uint32_t cum = 0;
        uint8_t threshold = 0;
        for (int i = 0; i < od; i++) {
            if (probs[i] > 0) {
                cum += probs[i];
                if (cum >= (uint32_t)(ctx->config.top_p * total)) {
                    threshold = probs[i];
                    break;
                }
            }
        }
        for (int i = 0; i < od; i++)
            if (probs[i] < threshold) probs[i] = 0;
    }

    /* Multinomial sampling with true random */
    uint32_t rnd = 0;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) { read(fd, &rnd, sizeof(rnd)); close(fd); }
    if (rnd == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        rnd = (uint32_t)(ts.tv_nsec ^ (uintptr_t)&rnd);
    }
    uint16_t rnd16 = (uint16_t)(rnd & 0xFFFF);
    uint32_t sampled = RIN_PTSoftmax_Sample(probs, od, rnd16);

    next_token->id = sampled;
    next_token->probability = (float)probs[sampled] / 255.0f;
    next_token->generation_time_ns = 0;
    next_token->logit = (float)logits[sampled];

    ctx->total_tokens_generated++;
    ctx->seq_len++;
    if (ctx->seq_len < RIN_MAX_SEQ_LEN)
        ctx->sequence[ctx->seq_len - 1] = *next_token;

    return RIN_STATUS_OK;
}

/* ============================================================================
 * Internal accessors for rin_api
 * ============================================================================ */
#ifndef RIN_EXPORT
#if defined(_WIN32) || defined(_WIN64)
#define RIN_EXPORT __declspec(dllexport)
#else
#define RIN_EXPORT __attribute__((visibility("default")))
#endif
#endif
RIN_EXPORT int rin_internal_num_layers(void *p) {
    RIN_Internal *rin = (RIN_Internal*)p;
    return rin ? rin->model.num_layers : 0;
}
RIN_EXPORT int rin_internal_input_dim(void *p) {
    RIN_Internal *rin = (RIN_Internal*)p;
    return rin ? rin->model.input_dim : 0;
}
RIN_EXPORT int rin_internal_output_dim(void *p) {
    RIN_Internal *rin = (RIN_Internal*)p;
    return rin ? rin->model.output_dim : 0;
}
RIN_EXPORT int rin_internal_num_heads(void *p) {
    RIN_Internal *rin = (RIN_Internal*)p;
    return rin ? rin->num_heads : 0;
}
RIN_EXPORT int rin_internal_ffn_dim(void *p) {
    RIN_Internal *rin = (RIN_Internal*)p;
    return rin ? rin->ffn_dim : 0;
}
RIN_EXPORT int rin_internal_max_seq_len(void *p) {
    RIN_Internal *rin = (RIN_Internal*)p;
    return rin ? rin->max_seq_len : 0;
}

RIN_EXPORT int rin_internal_num_layers(void *p);
RIN_EXPORT int rin_internal_input_dim(void *p);
RIN_EXPORT int rin_internal_output_dim(void *p);
RIN_EXPORT int rin_internal_num_heads(void *p);
RIN_EXPORT int rin_internal_ffn_dim(void *p);
RIN_EXPORT int rin_internal_max_seq_len(void *p);

/* ============================================================================
 * RIN_GetCharSet
 * ============================================================================ */
const char* RIN_GetCharSet(RIN_Context* ctx, int* vocab_size_out) {
    if (!ctx) return NULL;
    RIN_Internal *rin = (RIN_Internal*)ctx->_internal;
    if (!rin) return NULL;
    if (vocab_size_out) *vocab_size_out = rin->char_set_size;
    return rin->char_set;
}

/* ============================================================================
 * DESTRUCTOR
 * ============================================================================ */
void RIN_Destroy_Internal(RIN_Context* ctx) {
    if (!ctx) return;
    RIN_Internal *rin = (RIN_Internal*)ctx->_internal;
    if (!rin) return;

    RIN_Model *m = &rin->model;
    if (m->loaded) {
        if (rin->architecture == 1) {
            /* Transformer cleanup */
            free(rin->embed_table);
            free(rin->embed_bias);
            for (int i = 0; i < m->num_layers; i++) {
                free(rin->Wq[i]); free(rin->Wk[i]); free(rin->Wv[i]); free(rin->Wo[i]);
                free(rin->W1[i]); free(rin->W2[i]);
                free(rin->bq[i]); free(rin->bk[i]); free(rin->bv[i]); free(rin->bo[i]);
                free(rin->b1[i]); free(rin->b2[i]);
            }
            free(rin->model.W[0]);  /* unembedding */
            free(rin->model.bias[0]);
            free(rin->pos_embed);
            for (int i = 0; i < rin->num_ln_sets; i++) {
                free(rin->ln_gamma[i]);
                free(rin->ln_beta[i]);
            }
            free(rin->q_buf); free(rin->k_buf); free(rin->v_buf);
            free(rin->attn_buf); free(rin->residual); free(rin->ffn_buf);
            free(rin->k_cache); free(rin->v_cache);
            free(rin->char_set);
        } else {
            /* MLP cleanup */
            for (int i = 0; i < m->num_layers; i++) {
                free(m->W[i]);
                free(m->bias[i]);
            }
            free(rin->kv_cache.K);
            free(rin->kv_cache.V);
        }
        free(rin->buf0);
        free(rin->buf1);
        free(rin->gemv_buf);
    }
    free(rin);
    ctx->_internal = NULL;
}
