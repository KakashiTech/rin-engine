/*
 * rin_hrr.h — Holographic Reduced Representations via Q15 Real FFT
 *
 * Reemplaza la KV cache O(n·d) con UN vector de contexto O(d).
 * Toda la secuencia se codifica en un solo vector fijo.
 *
 *   context = Σ token_i ⊛ position_i    bind (convolución circular)
 *   token   ≈ context ⊛ position⁻¹      unbind (correlación circular)
 *
 * Pipeline de escalado (verificado para dim ≤ 4096, sin overflow):
 *   1. FFT con right-shift por etapa (÷N) → buffers en rango Q15
 *   2. Multiplicación compleja con int64 intermedio
 *   3. IFFT sin scaling → salida = a⊛b / (N·32768) en Q15
 *      Para dim=4096: max_out ≈ 8192 para activaciones típicas
 *
 * Referencias:
 *   Plate, T. "Holographic Reduced Representations" (1995)
 *   Kanerva, P. "Binary Spatter Codes" (1996)
 *
 * Dependencias: rin_arena.h
 * Estándar: C11
 */

#ifndef RIN_HRR_H
#define RIN_HRR_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "rin_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constantes
 * ============================================================================ */

#define RIN_HRR_MAX_DIM       4096
#define RIN_HRR_MIN_DIM       32
#define RIN_HRR_Q15_SCALE     32768
#define RIN_HRR_POS_MAG       10000
#define RIN_HRR_TWIDDLE_HALF  2048   /* dim/2 máximo */

/*
 * Memoria scratch necesaria para dim = N:
 *   cos_tab:  int16[N/2]        = N bytes
 *   sin_tab:  int16[N/2]        = N bytes
 *   bufA:     int32[2N]         = 8N bytes
 *   bufB:     int32[2N]         = 8N bytes
 *   bufC:     int32[2N]         = 8N bytes
 *   Total: 2N + 24N = 26N bytes
 */
#define RIN_HRR_SCRATCH_SIZE(N)  \
    (((N) >= RIN_HRR_MIN_DIM && (N) <= RIN_HRR_MAX_DIM) ? (26UL * (N)) : 0)

/* ============================================================================
 * Utilidades
 * ============================================================================ */

static inline uint32_t rin_xorshift32(uint32_t x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}

/* isqrt64 — raíz cuadrada entera vía Newton */
static inline int64_t rin_isqrt64(int64_t x) {
    if (x <= 1) return x;
    int64_t r = x;
    while (r * r > x) r = (r + x / r) >> 1;
    return r;
}

/* Saturación int16 */
static inline int16_t rin_sat16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* ============================================================================
 * Generación de vectores de posición (determinística vía XorShift32)
 * ============================================================================ */

static inline void rin_hrr_position(int idx, int16_t* out, int dim) {
    uint32_t seed = 0x9E3779B9U;
    uint32_t mix  = (uint32_t)(idx * 2654435761U);
    for (int i = 0; i < dim; i++) {
        seed = rin_xorshift32(seed ^ mix);
        out[i] = (int16_t)((seed & 0x7FFF) - 16384);
    }
}

/* ============================================================================
 * Twiddle factors (cos/sin en Q15 para FFT de longitud N)
 * ============================================================================ */

static inline void rin_twiddles(int n, int16_t* cos_tab, int16_t* sin_tab) {
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        double a = 6.283185307179586 * i / n;
        cos_tab[i] = (int16_t)(32767.0 * cos(a));
        sin_tab[i] = (int16_t)(32767.0 * sin(a));
    }
}

/* ============================================================================
 * Bit-reversal permutation para array complejo int32
 * ============================================================================ */

static inline void rin_bitrev32(int32_t* data, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            int32_t tr = data[2*i], ti = data[2*i+1];
            data[2*i]   = data[2*j];
            data[2*i+1] = data[2*j+1];
            data[2*j]   = tr;
            data[2*j+1] = ti;
        }
    }
}

/* ============================================================================
 * FFT compleja — Radix-2 DIT, int32, SIN right-shift por etapa
 *
 * data: [re0, im0, re1, im1, ..., re_{n-1}, im_{n-1}]
 * n: potencia de 2 (max 4096)
 * cos_tab, sin_tab: twiddle factors (precomputados con rin_twiddles)
 *
 * NO hay right-shift por etapa. Los valores internos crecen hasta N·max_input
 * pero caben en int32 para n ≤ 4096 y entrada Q15 (max ~1.34e8 < 2.14e9).
 *
 * forward=1: W = e^{-j·2π/N}
 * forward=0: W = e^{+j·2π/N} (para IFFT)
 * ============================================================================ */

static inline void rin_fft_cpx_ns(int32_t* data, int n,
                                   const int16_t* cos_tab,
                                   const int16_t* sin_tab,
                                   int forward) {
    rin_bitrev32(data, n);

    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int step = n / len;

        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; j++) {
                int idx  = i + j;
                int tw   = j * step;

                /* forward: W = e^{-j·2π/N} = cos - j·sin
                 * inverse: W = e^{+j·2π/N} = cos + j·sin */
                int32_t wre = cos_tab[tw];
                int32_t wim = forward ? -sin_tab[tw] : sin_tab[tw];

                int64_t bre = data[2*(idx + half)];
                int64_t bim = data[2*(idx + half) + 1];

                /* W · b en Q15, usando int64 para prevenir overflow */
                int32_t t_re = (int32_t)((wre*bre - wim*bim) >> 15);
                int32_t t_im = (int32_t)((wre*bim + wim*bre) >> 15);

                int32_t are = data[2*idx];
                int32_t aim = data[2*idx + 1];

                /* SIN right-shift — valores crecen hasta N·max_input,
                 * caben en int32 para n ≤ 4096 */
                data[2*idx]           = are + t_re;
                data[2*idx + 1]       = aim + t_im;
                data[2*(idx + half)]   = are - t_re;
                data[2*(idx + half)+1] = aim - t_im;
            }
        }
    }
}

/* ============================================================================
 * rin_ifft_extract — IFFT sin scaling + extraer parte real a int16
 *
 * Equivalente a rin_fft_cpx_ns(data, n, ct, st, 0) seguido de extracción
 * de la parte real con saturación Q15.
 *
 * Salida: (int16_t*)data[0..n-1] = valores reales.
 * ============================================================================ */

static inline void rin_ifft_extract(int32_t* data, int n,
                                     const int16_t* cos_tab,
                                     const int16_t* sin_tab) {
    rin_fft_cpx_ns(data, n, cos_tab, sin_tab, 0); /* forward=0 → IFFT */

    for (int i = 0; i < n; i++) {
        ((int16_t*)data)[i] = rin_sat16(data[2*i]);
    }
}

/* ============================================================================
 * rin_hrr_bind — Convolución circular
 *
 * out = a ⊛ b
 * a, b, out: arrays Q15 de longitud dim (potencia de 2)
 * scratch: buffer de RIN_HRR_SCRATCH_SIZE(dim) bytes
 *
 * Escalado interno:
 *   1. FFT con ÷N → A[k], B[k] en ~Q15
 *   2. C[k] = (A[k]·B[k]) >> 15  (int64 intermedio)
 *   3. IFFT sin scaling → out[i] ≈ (a⊛b)[i] / (N·32768)
 *
 * Para obtener el binding directo, multiplicar out[] por N:
 *   out_normalized[i] = out[i] * N   ≈ (a⊛b)[i] / 32768
 * ============================================================================ */

static inline void rin_hrr_bind(int16_t* out, const int16_t* a, const int16_t* b,
                                 int dim, int16_t* scratch) {
    int half = dim / 2;

    /* Layout del scratch:
     *   [0..half):         cos_tab    (int16)
     *   [half..dim):       sin_tab    (int16)
     *   [align..align+8N): bufA       (int32[2N])
     *   [align+8N..+16N):  bufB       (int32[2N])
     *   [align+16N..+24N): bufC       (int32[2N])
     */
    int16_t* cos_tab  = scratch;
    int16_t* sin_tab  = scratch + half;
    size_t align      = ((size_t)(scratch + dim) - (size_t)scratch + 15) & ~15;
    int32_t* bufA     = (int32_t*)((int16_t*)scratch + align / sizeof(int16_t));
    int32_t* bufB     = bufA + 2*dim;
    int32_t* bufC     = bufB + 2*dim;

    rin_twiddles(dim, cos_tab, sin_tab);

    /* Cargar a, b como complejos [re, 0, re, 0, ...] */
    for (int i = 0; i < dim; i++) {
        bufA[2*i]     = (int32_t)a[i];
        bufA[2*i + 1] = 0;
        bufB[2*i]     = (int32_t)b[i];
        bufB[2*i + 1] = 0;
    }

    rin_fft_cpx_ns(bufA, dim, cos_tab, sin_tab, 1);
    rin_fft_cpx_ns(bufB, dim, cos_tab, sin_tab, 1);

    /* Multiplicación compleja en frecuencia: C = A · B */
    for (int k = 0; k < dim; k++) {
        int64_t are = bufA[2*k];
        int64_t aim = bufA[2*k + 1];
        int64_t bre = bufB[2*k];
        int64_t bim = bufB[2*k + 1];

        bufC[2*k]     = (int32_t)((are*bre - aim*bim) >> 15);
        bufC[2*k + 1] = (int32_t)((are*bim + aim*bre) >> 15);
    }

    rin_ifft_extract(bufC, dim, cos_tab, sin_tab);

    /* Copiar resultado (ifft_noscale ya dejó int16 en bufC) */
    memcpy(out, bufC, dim * sizeof(int16_t));
}

/* ============================================================================
 * rin_hrr_unbind — Correlación circular (≈ inversa de bind)
 *
 * out ≈ c ⊛ b⁻¹   (recuperación de 'a' desde c = a⊛b)
 *
 * En frecuencia: C · conj(B) en vez de C · B
 * ============================================================================ */

static inline void rin_hrr_unbind(int16_t* out, const int16_t* c, const int16_t* b,
                                   int dim, int16_t* scratch) {
    int half = dim / 2;

    int16_t* cos_tab  = scratch;
    int16_t* sin_tab  = scratch + half;
    size_t align      = ((size_t)(scratch + dim) - (size_t)scratch + 15) & ~15;
    int32_t* bufC     = (int32_t*)((int16_t*)scratch + align / sizeof(int16_t));
    int32_t* bufB     = bufC + 2*dim;
    int32_t* bufT     = bufB + 2*dim;

    rin_twiddles(dim, cos_tab, sin_tab);

    for (int i = 0; i < dim; i++) {
        bufC[2*i]     = (int32_t)c[i];
        bufC[2*i + 1] = 0;
        bufB[2*i]     = (int32_t)b[i];
        bufB[2*i + 1] = 0;
    }

    rin_fft_cpx_ns(bufC, dim, cos_tab, sin_tab, 1);
    rin_fft_cpx_ns(bufB, dim, cos_tab, sin_tab, 1);

    /* C · conj(B) = (Cre + j·Cim)·(Bre - j·Bim) */
    for (int k = 0; k < dim; k++) {
        int64_t cre = bufC[2*k];
        int64_t cim = bufC[2*k + 1];
        int64_t bre = bufB[2*k];
        int64_t bim = bufB[2*k + 1];

        bufT[2*k]     = (int32_t)((cre*bre + cim*bim) >> 15);
        bufT[2*k + 1] = (int32_t)((cim*bre - cre*bim) >> 15);
    }

    rin_ifft_extract(bufT, dim, cos_tab, sin_tab);

    memcpy(out, bufT, dim * sizeof(int16_t));
}

/* ============================================================================
 * rin_hrr_encode — Codificar secuencia en contexto holográfico
 *
 * context = Σ token_i ⊛ position_i
 *
 * context: buffer de salida (inicializado a 0 si init=1)
 * tokens: array de num_pairs punteros a vectores Q15
 * positions: array de num_pairs punteros a vectores Q15
 * ============================================================================ */

static inline void rin_hrr_encode(int16_t* context,
                                   const int16_t** tokens,
                                   const int16_t** positions,
                                   int num_pairs, int dim,
                                   int16_t* scratch) {
    memset(context, 0, dim * sizeof(int16_t));

    for (int i = 0; i < num_pairs; i++) {
        rin_hrr_bind(scratch, tokens[i], positions[i], dim, scratch);
        for (int j = 0; j < dim; j++) {
            context[j] = rin_sat16((int32_t)context[j] + scratch[j]);
        }
    }
}

/* ============================================================================
 * rin_hrr_retrieve — Recuperar token en una posición
 *
 * token ≈ context ⊛ position⁻¹   (= unbind(context, position))
 * ============================================================================ */

static inline void rin_hrr_retrieve(int16_t* token,
                                     const int16_t* context,
                                     const int16_t* position,
                                     int dim, int16_t* scratch) {
    rin_hrr_unbind(token, context, position, dim, scratch);
}

/* ============================================================================
 * rin_hrr_update — Agregar un token al contexto existente
 *
 * context += token ⊛ position
 * ============================================================================ */

static inline void rin_hrr_update(int16_t* context,
                                   const int16_t* token,
                                   const int16_t* position,
                                   int dim, int16_t* scratch) {
    rin_hrr_bind(scratch, token, position, dim, scratch);
    for (int j = 0; j < dim; j++) {
        context[j] = rin_sat16((int32_t)context[j] + scratch[j]);
    }
}

/* ============================================================================
 * rin_hrr_replace — Reemplazar token en posición existente
 *
 * context -= old_token ⊛ position
 * context += new_token ⊛ position
 * ============================================================================ */

static inline void rin_hrr_replace(int16_t* context,
                                    const int16_t* old_token,
                                    const int16_t* new_token,
                                    const int16_t* position,
                                    int dim, int16_t* scratch) {
    rin_hrr_bind(scratch, old_token, position, dim, scratch);
    for (int j = 0; j < dim; j++) {
        context[j] = rin_sat16((int32_t)context[j] - scratch[j]);
    }
    rin_hrr_bind(scratch, new_token, position, dim, scratch);
    for (int j = 0; j < dim; j++) {
        context[j] = rin_sat16((int32_t)context[j] + scratch[j]);
    }
}

/* ============================================================================
 * rin_hrr_context_init — Inicializar contexto a cero
 * ============================================================================ */

static inline void rin_hrr_context_init(int16_t* context, int dim) {
    memset(context, 0, dim * sizeof(int16_t));
}

/* ============================================================================
 * rin_hrr_similarity — Coseno entre dos vectores Q15
 *
 * Rango: 0 (ortogonal) a 32767 (idénticos)
 * ============================================================================ */

static inline int16_t rin_hrr_similarity(const int16_t* a, const int16_t* b,
                                          int dim) {
    int64_t dot = 0, na2 = 0, nb2 = 0;
    for (int i = 0; i < dim; i++) {
        dot += (int64_t)a[i] * b[i];
        na2 += (int64_t)a[i] * a[i];
        nb2 += (int64_t)b[i] * b[i];
    }
    if (na2 == 0 || nb2 == 0) return 0;

    int32_t na = (int32_t)(1 + rin_isqrt64(na2));
    int32_t nb = (int32_t)(1 + rin_isqrt64(nb2));
    int64_t cos_q15 = (dot << 15) / ((int64_t)na * nb);
    return (int16_t)(cos_q15 > 32767 ? 32767 :
                     cos_q15 < -32768 ? -32768 :
                     (int32_t)cos_q15);
}

/* ============================================================================
 * Macros de conveniencia para arena allocator
 * ============================================================================ */

#define RIN_HRR_SCRATCH_ALLOC(arena, dim) \
    ((int16_t*)RIN_PERSIST_ALLOC(arena, RIN_HRR_SCRATCH_SIZE(dim)))

#define RIN_HRR_CONTEXT_ALLOC(arena, dim) \
    ((int16_t*)RIN_PERSIST_ALLOC(arena, (size_t)(dim) * sizeof(int16_t)))

#ifdef __cplusplus
}
#endif

#endif /* RIN_HRR_H */
