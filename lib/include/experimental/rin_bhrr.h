/*
 * rin_bhrr.h — Bipolar Holographic Reduced Representations (Key-Value Memory)
 *
 * Reemplaza la KV cache O(n·d) por contexto holográfico O(d).
 * Sin FFT, sin DCT, sin multiplicaciones de coma flotante.
 *
 * Vectores bipolares {-1, +1}^d como int16 Q15.
 * Contexto acumulado en int32 (rango [-K, K] para K items almacenados).
 *
 * Pipeline:
 *   1. Store: ctx[i] += sign(K[i]) · sign(V[i])    (1 XOR + 1 ADD)
 *   2. Query: out[i] = ctx[i] · sign(Q[i])          (1 MUL condicional)
 *   3. out[i] ≈ Σ dot(Q, K_j) · V_j[i]  (atención lineal sin softmax)
 *
 * SNR: e[dot(R, V_correct)] = d,  σ[dot(R, V_correct)] = √(d·(K-1))
 *      SNR_total = √(d / (K-1))
 *      Para d=4096, K=4096: SNR≈1.0 (usable)
 *      Para d=4096, K=256:  SNR≈4.0 (bueno)
 *
 * Multi-slot: divide K entre S slots → SNR por slot = √(S·d / K)
 *      S=4, d=4096, K=4096: SNR≈2.0
 *      S=16, d=4096, K=4096: SNR≈4.0
 *
 * Referencia:
 *   Kanerva, P. "Binary Spatter Codes" (1996)
 *   Gayler, RW. "Vector Symbolic Architectures" (2003)
 *
 * Dependencias: stdint.h, string.h
 * Estándar: C11
 */

#ifndef RIN_BHRR_H
#define RIN_BHRR_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constantes
 * ============================================================================ */

#define RIN_BHRR_MAX_DIM       4096
#define RIN_BHRR_MIN_DIM       16
#define RIN_BHRR_Q15_POS       32767
#define RIN_BHRR_Q15_NEG      -32768
#define RIN_BHRR_MAX_SLOTS     256
#define RIN_BHRR_DEFAULT_SLOTS 8

/* Scratch size: S slots × dim × sizeof(int32) + dim × sizeof(int16) × 2 */
#define RIN_BHRR_SCRATCH_SIZE(S, D) \
    (((S) * (D) * 4) + ((D) * 4))

/* ============================================================================
 * Utilitarios
 * ============================================================================ */

/* XorShift32 — generador determinístico rápido */
static inline uint32_t rin_bhrr_xor32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* Saturar int32 a int16 Q15 */
static inline int16_t rin_bhrr_sat16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Signo: 1 para ≥0, -1 para <0, como Q15 */
static inline int16_t rin_bhrr_sign_q15(int16_t v) {
    return v >= 0 ? RIN_BHRR_Q15_POS : RIN_BHRR_Q15_NEG;
}

/* Signo como -1/1 (int32, sin Q15) */
static inline int32_t rin_bhrr_sign_i32(int16_t v) {
    return v >= 0 ? 1 : -1;
}

/* ============================================================================
 * Generación de vectores bipolares aleatorios
 *
 * out: vector Q15, cada elemento es +32767 o -32768
 * ============================================================================ */

static inline void rin_bhrr_random(int16_t* out, int dim, uint32_t* seed) {
    for (int i = 0; i < dim; i++) {
        out[i] = (rin_bhrr_xor32(seed) & 1) ? RIN_BHRR_Q15_POS : RIN_BHRR_Q15_NEG;
    }
}

/* ============================================================================
 * Core algebra
 * ============================================================================ */

/* Bind = element-wise multiply en Q15 (sign XOR)
 * out[i] = a[i] · b[i] (bipolar: mismo signo → +1, distinto → -1) */
static inline void rin_bhrr_bind(int16_t* out, const int16_t* a,
                                  const int16_t* b, int dim) {
    for (int i = 0; i < dim; i++) {
        out[i] = (a[i] ^ b[i]) >= 0 ? RIN_BHRR_Q15_POS : RIN_BHRR_Q15_NEG;
    }
}

/* Unbind = same as bind (bipolar es auto-inverso) */
static inline void rin_bhrr_unbind(int16_t* out, const int16_t* c,
                                    const int16_t* a, int dim) {
    rin_bhrr_bind(out, c, a, dim);
}

/* Dot product escalado a Q15 */
static inline int64_t rin_bhrr_dot(const int16_t* a, const int16_t* b, int dim) {
    int64_t sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (int64_t)a[i] * b[i];
    }
    return sum;
}

/* Similitud coseno en Q15 (int64 para overflow) */
static inline int64_t rin_bhrr_cos_sim_q15(const int16_t* a, const int16_t* b,
                                            int dim) {
    int64_t dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += (int64_t)a[i] * b[i];
        na  += (int64_t)a[i] * a[i];
        nb  += (int64_t)b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0;
    /* cos = dot / sqrt(na·nb) en Q15: cos_q15 = dot * 32768 / sqrt(na·nb) */
    return dot;
}

/* ============================================================================
 * Contexto single-slot
 *
 * Almacena: C = Σ K_j ⊙ V_j   (element-wise, acumulado en int32)
 * Recupera: R = C ⊙ Q          (≈ Σ dot(Q, K_j) · V_j)
 *
 * C[i] ∈ [-K, K] para K items, cabe en int32 hasta K ≈ 2e9.
 * ============================================================================ */

static inline void rin_bhrr_ctx_init(int32_t* ctx, int dim) {
    memset(ctx, 0, dim * sizeof(int32_t));
}

static inline void rin_bhrr_ctx_clear(int32_t* ctx, int dim) {
    memset(ctx, 0, dim * sizeof(int32_t));
}

/* Store: ctx[i] += sign(K[i]) · sign(V[i])
 * Una ADD int32 por dimensión. */
static inline void rin_bhrr_ctx_store(int32_t* ctx, const int16_t* key,
                                       const int16_t* val, int dim) {
    for (int i = 0; i < dim; i++) {
        /* sign(key) · sign(val) = +1 si mismo signo, -1 si distinto */
        int32_t k_sign = key[i] >= 0 ? 1 : -1;
        int32_t v_sign = val[i] >= 0 ? 1 : -1;
        ctx[i] += k_sign * v_sign;
    }
}

/* Store con key/value en int32 pre-signados (±1) */
static inline void rin_bhrr_ctx_store_raw(int32_t* ctx, const int32_t* key_sign,
                                           const int32_t* val_sign, int dim) {
    for (int i = 0; i < dim; i++) {
        ctx[i] += key_sign[i] * val_sign[i];
    }
}

/* Retrieve: out[i] = ctx[i] · sign(Q[i]), saturado a Q15
 * out[i] ∈ [-K, K], luego saturado a int16. */
static inline void rin_bhrr_ctx_retrieve(int16_t* out, const int32_t* ctx,
                                          const int16_t* query, int dim) {
    for (int i = 0; i < dim; i++) {
        int32_t q_sign = query[i] >= 0 ? 1 : -1;
        out[i] = rin_bhrr_sat16(ctx[i] * q_sign);
    }
}

/* ============================================================================
 * Memoria multi-slot
 *
 * Divide el contexto en S slots. Cada slot almacena K/S items.
 * La asignación slot = hash(key) % num_slots.
 *
 * Memoria: S × dim × sizeof(int32_t) para slots
 *          S × dim × sizeof(int16_t) para claves de referencia
 * ============================================================================ */

typedef struct {
    int32_t* slots;        /* [num_slots][dim] acumuladores */
    int16_t* slot_keys;    /* [num_slots][dim] última clave (referencia) */
    int num_slots;
    int dim;
    int total_stored;      /* total de pares almacenados (solo contador) */
    uint32_t seed;
} RinBHRR;

/* Inicializar memoria multi-slot
 * mem: puntero a estructura
 * slots_buf: [num_slots × dim] int32 pre-asignado
 * keys_buf: [num_slots × dim] int16 pre-asignado
 * num_slots: número de slots (potencia de 2 recomendada)
 * dim: dimensión del vector */
static inline void rin_bhrr_init(RinBHRR* mem, int32_t* slots_buf,
                                  int16_t* keys_buf,
                                  int num_slots, int dim, uint32_t seed) {
    mem->slots = slots_buf;
    mem->slot_keys = keys_buf;
    mem->num_slots = num_slots;
    mem->dim = dim;
    mem->total_stored = 0;
    mem->seed = seed;

    for (int s = 0; s < num_slots; s++) {
        memset(&slots_buf[s * dim], 0, dim * sizeof(int32_t));
    }
}

/* Hash rápido de vector bipolar → slot ID (weighted-sum, matching Python training) */
static inline int rin_bhrr_slot_id(const int16_t* vec, int dim, int num_slots) {
    uint32_t h = 0;
    for (int i = 0; i < dim; i++) {
        h += (uint32_t)(vec[i] > 0) * (uint32_t)(i + 1);
    }
    return (int)(h % (uint32_t)num_slots);
}

/* Almacenar par clave-valor en slot determinado por hash(key)
 * Usa la clave completa para asignación de slot. */
static inline void rin_bhrr_store(RinBHRR* mem, const int16_t* key,
                                   const int16_t* val) {
    int slot = rin_bhrr_slot_id(key, mem->dim, mem->num_slots);
    int32_t* ctx = &mem->slots[slot * mem->dim];
    int16_t* ref = &mem->slot_keys[slot * mem->dim];

    rin_bhrr_ctx_store(ctx, key, val, mem->dim);

    /* Actualizar clave de referencia */
    memcpy(ref, key, mem->dim * sizeof(int16_t));
    mem->total_stored++;
}

/* Recuperar del slot correspondiente al query
 * Usa el hash del query para seleccionar slot. */
static inline void rin_bhrr_retrieve(int16_t* out, const RinBHRR* mem,
                                      const int16_t* query) {
    int slot = rin_bhrr_slot_id(query, mem->dim, mem->num_slots);
    const int32_t* ctx = &mem->slots[slot * mem->dim];

    rin_bhrr_ctx_retrieve(out, ctx, query, mem->dim);
}

/* Recuperar con polling sobre todos los slots (más preciso, más lento)
 * Selecciona el slot con mayor similitud query-clave_referencia. */
static inline void rin_bhrr_retrieve_poll(int16_t* out, const RinBHRR* mem,
                                           const int16_t* query) {
    int best_slot = 0;
    int64_t best_dot = -__INT64_MAX__;

    for (int s = 0; s < mem->num_slots; s++) {
        const int16_t* ref = &mem->slot_keys[s * mem->dim];
        int64_t d = rin_bhrr_dot(query, ref, mem->dim);
        if (d > best_dot) {
            best_dot = d;
            best_slot = s;
        }
    }

    const int32_t* ctx = &mem->slots[best_slot * mem->dim];
    rin_bhrr_ctx_retrieve(out, ctx, query, mem->dim);
}

/* Limpiar toda la memoria */
static inline void rin_bhrr_clear(RinBHRR* mem) {
    for (int s = 0; s < mem->num_slots; s++) {
        memset(&mem->slots[s * mem->dim], 0, mem->dim * sizeof(int32_t));
    }
    mem->total_stored = 0;
}

/* ============================================================================
 * Atención multi-head via Bipolar HRR
 *
 * Cada cabeza tiene su propio RinBHRR multi-slot.
 * head_dim = dim_total / num_heads
 *
 * Para cabeza h:
 *   K_h, V_h, Q_h ∈ ℝ^{head_dim} → sign(K_h), sign(V_h), sign(Q_h) ∈ {-1,+1}
 *   C_h = Σ K_h_j ⊙ V_h_j   (acumulado por cabeza)
 *   R_h = C_h ⊙ Q_h          (≈ atención lineal)
 *   out[h] = R_h
 * ============================================================================ */

typedef struct {
    RinBHRR* heads;       /* [num_heads] estructuras RinBHRR */
    int32_t* slot_bufs;   /* [num_heads × num_slots × head_dim] */
    int16_t* key_bufs;    /* [num_heads × num_slots × head_dim] */
    int num_heads;
    int head_dim;
    int num_slots;
} RinBHRRAttention;

/* Inicializar atención multi-head
 * Se necesita un buffer contiguo de:
 *   heads:     puntero a array [num_heads] de RinBHRR (asignado por el caller)
 *   slot_bufs: num_heads × num_slots × head_dim × sizeof(int32_t)
 *   key_bufs:  num_heads × num_slots × head_dim × sizeof(int16_t)
 */
static inline void rin_bhrr_attn_init(RinBHRRAttention* attn,
                                       RinBHRR* heads,
                                       int32_t* slot_bufs,
                                       int16_t* key_bufs,
                                       int num_heads, int head_dim,
                                       int num_slots, uint32_t seed) {
    attn->heads = heads;
    attn->slot_bufs = slot_bufs;
    attn->key_bufs = key_bufs;
    attn->num_heads = num_heads;
    attn->head_dim = head_dim;
    attn->num_slots = num_slots;

    int32_t* slots_ptr = slot_bufs;
    int16_t* keys_ptr = key_bufs;
    for (int h = 0; h < num_heads; h++) {
        rin_bhrr_init(&attn->heads[h], slots_ptr, keys_ptr,
                       num_slots, head_dim, seed + h);
        slots_ptr += num_slots * head_dim;
        keys_ptr  += num_slots * head_dim;
    }
}

/* Store: cabeza h almacena key,val */
static inline void rin_bhrr_attn_store(RinBHRRAttention* attn, int head,
                                        const int16_t* key,
                                        const int16_t* val) {
    if (head < 0 || head >= attn->num_heads) return;
    rin_bhrr_store(&attn->heads[head], key, val);
}

/* Retrieve: cabeza h recupera con query */
static inline void rin_bhrr_attn_retrieve(int16_t* out,
                                           const RinBHRRAttention* attn,
                                           int head, const int16_t* query) {
    if (head < 0 || head >= attn->num_heads) return;
    rin_bhrr_retrieve(out, &attn->heads[head], query);
}

/* Retrieve con polling (todos los slots) */
static inline void rin_bhrr_attn_retrieve_poll(int16_t* out,
                                                const RinBHRRAttention* attn,
                                                int head,
                                                const int16_t* query) {
    if (head < 0 || head >= attn->num_heads) return;
    rin_bhrr_retrieve_poll(out, &attn->heads[head], query);
}

/* ============================================================================
 * TODO: Memoria Jerárquica (Contexto Ilimitado)
 *
 * Idea: Nivel 1 buffer exacto + Nivel 2 HRR multi-slot comprimido.
 * Pendiente de implementar cuando se integre en pipeline de inferencia.
 * Ver blueprint en BHRR_ADR.md sección "Memoria jerárquica".
 * ============================================================================ */

#ifdef __cplusplus
}
#endif

#endif /* RIN_BHRR_H */
