#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_int(const char *s, int *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0') return 0;
    if (v < -2147483648L || v > 2147483647L) return 0;
    *out = (int)v;
    return 1;
}

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void print_usage(const char *argv0) {
    printf("Usage: %s [x_sparsity_pct x_noise_thr y_noise_thr_msb refine_thr_msb]\n", argv0);
    printf("Defaults: 70 8 64 256\n");
}

#define IN_DIM 512
#define OUT_DIM 512

#define RINX_LUT_SIZE 2048
#define RINX_ENERGY_BINS 8

// Primary LUT: indexed by (energy_bin, msb_bin_primary)
#define RINX_MSB_SCORE_MAX 32768
#define RINX_MSB_BINS_PRIMARY (RINX_LUT_SIZE / RINX_ENERGY_BINS)  // 256
#define RINX_MSB_BIN_WIDTH_PRIMARY (2 * RINX_MSB_SCORE_MAX / RINX_MSB_BINS_PRIMARY)

// Secondary LUT (penumbra): indexed by (energy_bin, probe_bin, msb_bin_secondary)
#define RINX_PROBE_DIM 64
#define RINX_PROBE_BINS 4
#define RINX_MSB_BINS_SECONDARY (RINX_LUT_SIZE / (RINX_ENERGY_BINS * RINX_PROBE_BINS)) // 64
#define RINX_MSB_BIN_WIDTH_SECONDARY (2 * RINX_MSB_SCORE_MAX / RINX_MSB_BINS_SECONDARY)

static float g_res_lut_mean[RINX_LUT_SIZE] __attribute__((aligned(64)));
static float g_res_lut_var[RINX_LUT_SIZE] __attribute__((aligned(64)));
static uint32_t g_res_lut_count[RINX_LUT_SIZE] __attribute__((aligned(64)));

static float g_res2_lut_mean[RINX_LUT_SIZE] __attribute__((aligned(64)));
static float g_res2_lut_var[RINX_LUT_SIZE] __attribute__((aligned(64)));
static uint32_t g_res2_lut_count[RINX_LUT_SIZE] __attribute__((aligned(64)));

static inline int rinx_msb_bin_primary(int32_t msb_score) {
    const int32_t msb_cl = clamp_i32(msb_score, -RINX_MSB_SCORE_MAX, RINX_MSB_SCORE_MAX - 1);
    const int32_t shifted = msb_cl + RINX_MSB_SCORE_MAX;
    int b = (int)(shifted / RINX_MSB_BIN_WIDTH_PRIMARY);
    if (b < 0) b = 0;
    if (b >= RINX_MSB_BINS_PRIMARY) b = RINX_MSB_BINS_PRIMARY - 1;
    return b;
}

static inline int rinx_msb_bin_secondary(int32_t msb_score) {
    const int32_t msb_cl = clamp_i32(msb_score, -RINX_MSB_SCORE_MAX, RINX_MSB_SCORE_MAX - 1);
    const int32_t shifted = msb_cl + RINX_MSB_SCORE_MAX;
    int b = (int)(shifted / RINX_MSB_BIN_WIDTH_SECONDARY);
    if (b < 0) b = 0;
    if (b >= RINX_MSB_BINS_SECONDARY) b = RINX_MSB_BINS_SECONDARY - 1;
    return b;
}

static inline int rinx_energy_bin_from_active(int active_count) {
    // Map [0..IN_DIM] into RINX_ENERGY_BINS bins.
    // Using active_count keeps this cheap and stable.
    int scaled = (active_count * RINX_ENERGY_BINS) / IN_DIM;
    if (scaled < 0) scaled = 0;
    if (scaled >= RINX_ENERGY_BINS) scaled = RINX_ENERGY_BINS - 1;
    return scaled;
}

static inline int rinx_probe_bin_from_score(int32_t probe_score) {
    // probe_score is a cheap partial dot; bin by magnitude
    int32_t a = probe_score;
    if (a < 0) a = -a;
    if (a < 256) return 0;
    if (a < 1024) return 1;
    if (a < 4096) return 2;
    return 3;
}

static inline int rinx_lut_primary_index(int32_t msb_score, int energy_bin) {
    if (energy_bin < 0) energy_bin = 0;
    if (energy_bin >= RINX_ENERGY_BINS) energy_bin = RINX_ENERGY_BINS - 1;
    const int mb = rinx_msb_bin_primary(msb_score);
    return energy_bin * RINX_MSB_BINS_PRIMARY + mb;
}

static inline int rinx_lut_secondary_index(int32_t msb_score, int energy_bin, int probe_bin) {
    if (energy_bin < 0) energy_bin = 0;
    if (energy_bin >= RINX_ENERGY_BINS) energy_bin = RINX_ENERGY_BINS - 1;
    if (probe_bin < 0) probe_bin = 0;
    if (probe_bin >= RINX_PROBE_BINS) probe_bin = RINX_PROBE_BINS - 1;
    const int mb = rinx_msb_bin_secondary(msb_score);
    return (energy_bin * RINX_PROBE_BINS + probe_bin) * RINX_MSB_BINS_SECONDARY + mb;
}

static inline uint64_t rdtsc_start(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned hi, lo;
    __asm__ __volatile__(
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
        : "a"(0)
        : "%rbx", "%rcx");
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

static inline uint64_t rdtsc_end(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned hi, lo;
    __asm__ __volatile__(
        "rdtscp\n\t"
        : "=a"(lo), "=d"(hi)
        :
        : "%rbx", "%rcx");
    __asm__ __volatile__("cpuid\n\t" : : "a"(0) : "%rbx", "%rcx", "%rdx");
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

static inline int8_t sat_int8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline int popcount_u32(uint32_t x) {
    return __builtin_popcount(x);
}

typedef struct {
    int8_t *w;
    float w_scale;
    float x_scale;
} rinx_resonant_layer_t;

static int build_sparse_x(const int8_t *x_int8, int8_t *x_sparse, int8_t x_noise_thr) {
    uint32_t active_mask_words[(IN_DIM + 31) / 32];
    memset(active_mask_words, 0, sizeof(active_mask_words));

    int active_count = 0;
    for (int i = 0; i < IN_DIM; i++) {
        int8_t v = x_int8[i];
        if (v >= x_noise_thr || v <= (int8_t)-x_noise_thr) {
            active_mask_words[i >> 5] |= (uint32_t)1u << (i & 31);
            active_count++;
        }
    }

    memcpy(x_sparse, x_int8, IN_DIM);
    for (int w = 0; w < (IN_DIM + 31) / 32; w++) {
        uint32_t m = active_mask_words[w];
        if (m == 0) {
            memset(&x_sparse[w * 32], 0, 32);
            continue;
        }
        if (m == 0xFFFFFFFFu) continue;
        int base = w * 32;
        for (int b = 0; b < 32 && (base + b) < IN_DIM; b++) {
            if (((m >> b) & 1u) == 0u) x_sparse[base + b] = 0;
        }
    }

    return active_count;
}

static void init_layer(rinx_resonant_layer_t *layer) {
    layer->w = (int8_t *)aligned_alloc(32, (size_t)OUT_DIM * (size_t)IN_DIM);
    if (!layer->w) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    layer->w_scale = 1.0f / 127.0f;
    layer->x_scale = 1.0f / 127.0f;

    for (int i = 0; i < OUT_DIM * IN_DIM; i++) {
        int v = (rand() % 255) - 127;
        layer->w[i] = (int8_t)v;
    }
}

static void free_layer(rinx_resonant_layer_t *layer) {
    free(layer->w);
    layer->w = NULL;
}

static inline int32_t dot_int8_avx2(const int8_t *w, const int8_t *x, int cols) {
    __m256i acc = _mm256_setzero_si256();

    int j = 0;
    for (; j + 32 <= cols; j += 32) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + j));
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + j));

        __m128i w_lo_128 = _mm256_extracti128_si256(wv, 0);
        __m128i w_hi_128 = _mm256_extracti128_si256(wv, 1);
        __m128i x_lo_128 = _mm256_extracti128_si256(xv, 0);
        __m128i x_hi_128 = _mm256_extracti128_si256(xv, 1);

        __m256i w_lo = _mm256_cvtepi8_epi16(w_lo_128);
        __m256i w_hi = _mm256_cvtepi8_epi16(w_hi_128);
        __m256i x_lo = _mm256_cvtepi8_epi16(x_lo_128);
        __m256i x_hi = _mm256_cvtepi8_epi16(x_hi_128);

        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(w_lo, x_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(w_hi, x_hi));
    }

    __m128i lo = _mm256_extracti128_si256(acc, 0);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t s = _mm_extract_epi32(sum, 0);

    for (; j < cols; j++) {
        s += (int32_t)w[j] * (int32_t)x[j];
    }

    return s;
}

static inline int32_t dot_msb4_int8_avx2(const int8_t *w, const int8_t *x, int cols) {
    __m256i acc = _mm256_setzero_si256();

    int j = 0;
    for (; j + 32 <= cols; j += 32) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + j));
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + j));

        __m128i w_lo_128 = _mm256_extracti128_si256(wv, 0);
        __m128i w_hi_128 = _mm256_extracti128_si256(wv, 1);
        __m128i x_lo_128 = _mm256_extracti128_si256(xv, 0);
        __m128i x_hi_128 = _mm256_extracti128_si256(xv, 1);

        __m256i w_lo16 = _mm256_cvtepi8_epi16(w_lo_128);
        __m256i w_hi16 = _mm256_cvtepi8_epi16(w_hi_128);
        __m256i x_lo16 = _mm256_cvtepi8_epi16(x_lo_128);
        __m256i x_hi16 = _mm256_cvtepi8_epi16(x_hi_128);

        w_lo16 = _mm256_srai_epi16(w_lo16, 4);
        w_hi16 = _mm256_srai_epi16(w_hi16, 4);
        x_lo16 = _mm256_srai_epi16(x_lo16, 4);
        x_hi16 = _mm256_srai_epi16(x_hi16, 4);

        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(w_lo16, x_lo16));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(w_hi16, x_hi16));
    }

    __m128i lo = _mm256_extracti128_si256(acc, 0);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t s = _mm_extract_epi32(sum, 0);

    for (; j < cols; j++) {
        s += (int32_t)(w[j] >> 4) * (int32_t)(x[j] >> 4);
    }

    return s;
}

typedef struct {
    uint64_t skipped_neurons;
    uint64_t refined_neurons;
    uint64_t total_neurons;
    uint64_t active_inputs;
    uint64_t total_inputs;
    uint64_t clipped_outputs;
    uint64_t lut_trusted;
    uint64_t lut_penumbra;
} rinx_resonant_stats_t;

typedef struct {
    int x_sparsity_pct;
    int x_noise_thr;
    int y_noise_thr_msb;
    int refine_thr_msb;
    float penumbra_var_thr;
    int sat_thr_msb;
} rinx_resonant_config_t;

static void rinx_calibrate_resonance(const rinx_resonant_layer_t *layer,
                                     const int8_t *x_int8,
                                     int8_t x_noise_thr,
                                     int32_t y_noise_thr_msb) {
    // Build LUT: for each msb_score bin, store mean/var of exact resonant output
    // Reference is exact dot on x_sparse, with same skip rule.
    memset(g_res_lut_mean, 0, sizeof(g_res_lut_mean));
    memset(g_res_lut_var, 0, sizeof(g_res_lut_var));
    memset(g_res_lut_count, 0, sizeof(g_res_lut_count));
    memset(g_res2_lut_mean, 0, sizeof(g_res2_lut_mean));
    memset(g_res2_lut_var, 0, sizeof(g_res2_lut_var));
    memset(g_res2_lut_count, 0, sizeof(g_res2_lut_count));

    // Estimate input sparsity from provided x (ratio of zeros)
    int zeros = 0;
    for (int i = 0; i < IN_DIM; i++) if (x_int8[i] == 0) zeros++;
    const int sparsity_pct = (zeros * 100) / IN_DIM;

    const float scale = layer->w_scale * layer->x_scale;

    // Calibrate with multiple random inputs to stabilize mean/var per bin
    // This is essential because msb_score alone is not a sufficient statistic.
    const int CALIB_SAMPLES = 24;
    int8_t x_cal[IN_DIM];
    int8_t x_sparse[IN_DIM];

    for (int s = 0; s < CALIB_SAMPLES; s++) {
        for (int i = 0; i < IN_DIM; i++) {
            int v = (rand() % 255) - 127;
            if ((rand() % 100) < sparsity_pct) v = 0;
            x_cal[i] = (int8_t)v;
        }

        const int active_count = build_sparse_x(x_cal, x_sparse, x_noise_thr);
        const int ebin = rinx_energy_bin_from_active(active_count);

        // Welford online variance per LUT cell
        for (int r = 0; r < OUT_DIM; r++) {
            const int8_t *wrow = layer->w + (size_t)r * (size_t)IN_DIM;
            const int32_t msb_score = dot_msb4_int8_avx2(wrow, x_sparse, IN_DIM);
            const int32_t probe_score = dot_int8_avx2(wrow, x_sparse, RINX_PROBE_DIM);
            const int pbin = rinx_probe_bin_from_score(probe_score);
            float y = 0.0f;
            if (!(msb_score >= -y_noise_thr_msb && msb_score <= y_noise_thr_msb)) {
                const int32_t acc = dot_int8_avx2(wrow, x_sparse, IN_DIM);
                y = (float)acc * scale;
            }

            const int idx = rinx_lut_primary_index(msb_score, ebin);
            uint32_t c = g_res_lut_count[idx] + 1;
            g_res_lut_count[idx] = c;

            float delta = y - g_res_lut_mean[idx];
            g_res_lut_mean[idx] += delta / (float)c;
            float delta2 = y - g_res_lut_mean[idx];
            g_res_lut_var[idx] += delta * delta2; // accum M2

            const int idx2 = rinx_lut_secondary_index(msb_score, ebin, pbin);
            uint32_t c2 = g_res2_lut_count[idx2] + 1;
            g_res2_lut_count[idx2] = c2;
            float d1 = y - g_res2_lut_mean[idx2];
            g_res2_lut_mean[idx2] += d1 / (float)c2;
            float d2 = y - g_res2_lut_mean[idx2];
            g_res2_lut_var[idx2] += d1 * d2;
        }
    }

    for (int i = 0; i < RINX_LUT_SIZE; i++) {
        uint32_t c = g_res_lut_count[i];
        if (c > 1) g_res_lut_var[i] = g_res_lut_var[i] / (float)(c - 1);
        else g_res_lut_var[i] = 0.0f;

        uint32_t c2 = g_res2_lut_count[i];
        if (c2 > 1) g_res2_lut_var[i] = g_res2_lut_var[i] / (float)(c2 - 1);
        else g_res2_lut_var[i] = 0.0f;
    }
}

static void rinx_resonant_infer(const rinx_resonant_layer_t *layer,
                                const int8_t *x_int8,
                                int8_t *y_int8,
                                float *y_fp32_out,
                                int8_t x_noise_thr,
                                int32_t y_noise_thr_msb,
                                int32_t refine_thr_msb,
                                float penumbra_var_thr,
                                int32_t sat_thr_msb,
                                rinx_resonant_stats_t *stats) {
    int8_t x_sparse[IN_DIM];
    const int active_count = build_sparse_x(x_int8, x_sparse, x_noise_thr);
    const int ebin = rinx_energy_bin_from_active(active_count);
    stats->active_inputs += (uint64_t)active_count;
    stats->total_inputs += (uint64_t)IN_DIM;

    for (int r = 0; r < OUT_DIM; r++) {
        const int8_t *wrow = layer->w + (size_t)r * (size_t)IN_DIM;

        // Bit-plane score (MSB4 only) for gating + LUT index
        const int32_t msb_score = dot_msb4_int8_avx2(wrow, x_sparse, IN_DIM);

        stats->total_neurons++;

        if (msb_score >= -y_noise_thr_msb && msb_score <= y_noise_thr_msb) {
            y_int8[r] = 0;
            if (y_fp32_out) y_fp32_out[r] = 0.0f;
            stats->skipped_neurons++;
            continue;
        }

        const int idx = rinx_lut_primary_index(msb_score, ebin);
        const float y_lut = g_res_lut_mean[idx];
        const float v_lut = g_res_lut_var[idx];

        // Non-linear gating:
        // - Saturated scores: trust LUT (no refine)
        // - Penumbra (high variance): refine only if near decision boundary
        // - Otherwise trust LUT
        float y_f;
        int do_refine = 0;

        // Special mode: penumbra_var_thr < 0 forces refine for all non-skipped neurons
        if (penumbra_var_thr < 0.0f) do_refine = 1;

        if (!do_refine && sat_thr_msb > 0 && (msb_score >= sat_thr_msb || msb_score <= -sat_thr_msb)) {
            stats->lut_trusted++;
        } else if (!do_refine && v_lut > penumbra_var_thr) {
            stats->lut_penumbra++;
            // Penumbra stage: compute cheap probe, then consult secondary LUT
            const int32_t probe_score = dot_int8_avx2(wrow, x_sparse, RINX_PROBE_DIM);
            const int pbin = rinx_probe_bin_from_score(probe_score);
            const int idx2 = rinx_lut_secondary_index(msb_score, ebin, pbin);
            const float y2 = g_res2_lut_mean[idx2];
            const float v2 = g_res2_lut_var[idx2];

            if (v2 <= penumbra_var_thr) {
                y_f = y2;
                do_refine = 0;
                // override trusted path
            } else {
                if (msb_score >= -refine_thr_msb && msb_score <= refine_thr_msb) do_refine = 1;
                else {
                    // Still uncertain, but outside refine band: trust y2 anyway
                    y_f = y2;
                    do_refine = 0;
                }
            }
        } else {
            stats->lut_trusted++;
        }

        if (do_refine) {
            const int32_t acc = dot_int8_avx2(wrow, x_sparse, IN_DIM);
            y_f = (float)acc * (layer->w_scale * layer->x_scale);
            stats->refined_neurons++;
        } else {
            // If y_f not set by secondary stage, use primary LUT
            if (stats->lut_penumbra == 0 || !(v_lut > penumbra_var_thr)) y_f = y_lut;
        }

        if (y_fp32_out) y_fp32_out[r] = y_f;

        int yq = (int)lrintf(y_f);
        int8_t ys = sat_int8(yq);
        y_int8[r] = ys;
        if ((yq > 127) || (yq < -128)) stats->clipped_outputs++;
    }
}

static void reference_resonant_exact(const rinx_resonant_layer_t *layer,
                                     const int8_t *x_int8,
                                     float *y_ref,
                                     int8_t x_noise_thr,
                                     int32_t y_noise_thr_msb) {
    int8_t x_sparse[IN_DIM];
    (void)build_sparse_x(x_int8, x_sparse, x_noise_thr);

    const float scale = layer->w_scale * layer->x_scale;
    for (int r = 0; r < OUT_DIM; r++) {
        const int8_t *wrow = layer->w + (size_t)r * (size_t)IN_DIM;
        const int32_t msb_score = dot_msb4_int8_avx2(wrow, x_sparse, IN_DIM);
        if (msb_score >= -y_noise_thr_msb && msb_score <= y_noise_thr_msb) {
            y_ref[r] = 0.0f;
            continue;
        }
        const int32_t acc = dot_int8_avx2(wrow, x_sparse, IN_DIM);
        y_ref[r] = (float)acc * scale;
    }
}

static void reference_dense_exact(const rinx_resonant_layer_t *layer,
                                  const int8_t *x_int8,
                                  float *y_ref_dense) {
    const float scale = layer->w_scale * layer->x_scale;
    for (int r = 0; r < OUT_DIM; r++) {
        const int8_t *wrow = layer->w + (size_t)r * (size_t)IN_DIM;
        const int32_t acc = dot_int8_avx2(wrow, x_int8, IN_DIM);
        y_ref_dense[r] = (float)acc * scale;
    }
}

static void compute_fidelity_norm(const float *y_est, const float *y_ref, int n,
                                  double *out_norm, double *out_ref_std, double *out_err_std, double *out_mae) {
    double mean = 0.0;
    double mae = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)y_est[i] - (double)y_ref[i];
        mean += d;
        mae += fabs(d);
    }
    mean /= (double)n;
    mae /= (double)n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = ((double)y_est[i] - (double)y_ref[i]) - mean;
        var += d * d;
    }
    var /= (double)n;
    double err_std = sqrt(var);
    if (!isfinite(err_std)) err_std = -1.0;

    double ref_mean = 0.0;
    for (int i = 0; i < n; i++) ref_mean += (double)y_ref[i];
    ref_mean /= (double)n;

    double ref_var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = ((double)y_ref[i]) - ref_mean;
        ref_var += d * d;
    }
    ref_var /= (double)n;
    double ref_std = sqrt(ref_var);
    if (!isfinite(ref_std)) ref_std = -1.0;

    double norm = (ref_std > 0.0) ? (err_std / ref_std) : -1.0;
    *out_norm = norm;
    *out_ref_std = ref_std;
    *out_err_std = err_std;
    *out_mae = mae;
}

typedef struct {
    rinx_resonant_config_t cfg;
    double cycles_per_inf;
    double skip_pct;
    double refine_pct;
    double fidelity_norm;
    double onnx_gain_est;
} rinx_result_row_t;

static rinx_result_row_t run_one_config(const rinx_resonant_layer_t *layer,
                                        const int8_t *x_int8,
                                        const rinx_resonant_config_t *cfg,
                                        const float *y_ref_dense) {
    rinx_calibrate_resonance(layer, x_int8, (int8_t)cfg->x_noise_thr, cfg->y_noise_thr_msb);

    int8_t y_int8[OUT_DIM];
    float y_est[OUT_DIM];

    const int warmup = 100;
    const int iters = 2000;
    rinx_resonant_stats_t stats = {0};

    for (int i = 0; i < warmup; i++) {
        rinx_resonant_infer(layer, x_int8, y_int8, y_est,
                            (int8_t)cfg->x_noise_thr,
                            cfg->y_noise_thr_msb,
                            cfg->refine_thr_msb,
                            cfg->penumbra_var_thr,
                            cfg->sat_thr_msb,
                            &stats);
    }

    stats = (rinx_resonant_stats_t){0};
    uint64_t t0 = rdtsc_start();
    for (int i = 0; i < iters; i++) {
        rinx_resonant_infer(layer, x_int8, y_int8, y_est,
                            (int8_t)cfg->x_noise_thr,
                            cfg->y_noise_thr_msb,
                            cfg->refine_thr_msb,
                            cfg->penumbra_var_thr,
                            cfg->sat_thr_msb,
                            &stats);
    }
    uint64_t t1 = rdtsc_end();

    double fidelity_norm, ref_std, err_std, mae;
    compute_fidelity_norm(y_est, y_ref_dense, OUT_DIM, &fidelity_norm, &ref_std, &err_std, &mae);

    double cycles_per_inf = (double)(t1 - t0) / (double)iters;
    double skip_pct = 100.0 * (double)stats.skipped_neurons / (double)stats.total_neurons;
    double refine_pct = 100.0 * (double)stats.refined_neurons / (double)stats.total_neurons;

    const double cpu_ghz = 3.2;
    const double cycles_onnx = 0.0159e-3 * cpu_ghz * 1e9;
    double gain = cycles_onnx / cycles_per_inf;

    rinx_result_row_t row;
    row.cfg = *cfg;
    row.cycles_per_inf = cycles_per_inf;
    row.skip_pct = skip_pct;
    row.refine_pct = refine_pct;
    row.fidelity_norm = fidelity_norm;
    row.onnx_gain_est = gain;
    return row;
}

static int dominates(const rinx_result_row_t *a, const rinx_result_row_t *b) {
    int better_or_equal = 1;
    int strictly_better = 0;

    if (a->cycles_per_inf > b->cycles_per_inf) better_or_equal = 0;
    else if (a->cycles_per_inf < b->cycles_per_inf) strictly_better = 1;

    if (a->skip_pct < b->skip_pct) better_or_equal = 0;
    else if (a->skip_pct > b->skip_pct) strictly_better = 1;

    if (a->refine_pct > b->refine_pct) better_or_equal = 0;
    else if (a->refine_pct < b->refine_pct) strictly_better = 1;

    return better_or_equal && strictly_better;
}

static int rinx_autotune_resonance(const rinx_resonant_layer_t *layer,
                                  const int8_t *x_int8,
                                  float fidelity_target_norm,
                                  rinx_result_row_t *out_rows,
                                  int max_rows) {
    const int x_noise_list[] = {0, 4, 8, 12, 16};
    const int y_noise_list[] = {0, 32, 64, 96, 128};
    const int refine_list[] = {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096};
    const float var_list[] = {-1.0f, 0.0f, 1e-6f, 1e-4f, 1e-3f, 1e-2f, 5e-2f, 1e-1f, 2e-1f};

    float y_ref_dense[OUT_DIM];
    reference_dense_exact(layer, x_int8, y_ref_dense);

    int count = 0;
    for (int xi = 0; xi < (int)(sizeof(x_noise_list) / sizeof(x_noise_list[0])); xi++) {
        for (int yi = 0; yi < (int)(sizeof(y_noise_list) / sizeof(y_noise_list[0])); yi++) {
            for (int ri = 0; ri < (int)(sizeof(refine_list) / sizeof(refine_list[0])); ri++) {
                for (int vi = 0; vi < (int)(sizeof(var_list) / sizeof(var_list[0])); vi++) {
                    rinx_resonant_config_t cfg;
                    cfg.x_sparsity_pct = 0;
                    cfg.x_noise_thr = x_noise_list[xi];
                    cfg.y_noise_thr_msb = y_noise_list[yi];
                    cfg.refine_thr_msb = refine_list[ri];
                    cfg.penumbra_var_thr = var_list[vi];
                    cfg.sat_thr_msb = 512;

                    rinx_result_row_t row = run_one_config(layer, x_int8, &cfg, y_ref_dense);
                    if (row.fidelity_norm >= 0.0 && row.fidelity_norm <= (double)fidelity_target_norm) {
                        if (count < max_rows) out_rows[count] = row;
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

int main(int argc, char **argv) {
    srand(42);

    int x_sparsity_pct = 70;
    int x_noise_thr = 8;
    int y_noise_thr_msb = 64;
    int refine_thr_msb = 256;

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc == 5) {
        if (!parse_int(argv[1], &x_sparsity_pct) ||
            !parse_int(argv[2], &x_noise_thr) ||
            !parse_int(argv[3], &y_noise_thr_msb) ||
            !parse_int(argv[4], &refine_thr_msb)) {
            print_usage(argv[0]);
            return 2;
        }
    } else if (argc != 1) {
        print_usage(argv[0]);
        return 2;
    }

    if (x_sparsity_pct < 0) x_sparsity_pct = 0;
    if (x_sparsity_pct > 100) x_sparsity_pct = 100;

    if (x_noise_thr < 0) x_noise_thr = 0;
    if (x_noise_thr > 127) x_noise_thr = 127;
    if (y_noise_thr_msb < 0) y_noise_thr_msb = 0;
    if (refine_thr_msb < 0) refine_thr_msb = 0;

    rinx_resonant_layer_t layer;
    init_layer(&layer);

    int8_t x[IN_DIM];
    for (int i = 0; i < IN_DIM; i++) {
        int v = (rand() % 255) - 127;
        if ((rand() % 100) < x_sparsity_pct) v = 0;
        x[i] = (int8_t)v;
    }

    // === Auto-tuner (Pareto front) ===
    // Baseline run (manual params) + tuned configs under fidelity target.
    float y_ref_dense[OUT_DIM];
    reference_dense_exact(&layer, x, y_ref_dense);

    rinx_resonant_config_t base;
    base.x_sparsity_pct = x_sparsity_pct;
    base.x_noise_thr = x_noise_thr;
    base.y_noise_thr_msb = y_noise_thr_msb;
    base.refine_thr_msb = refine_thr_msb;
    base.penumbra_var_thr = 0.05f;
    base.sat_thr_msb = 512;

    rinx_result_row_t base_row = run_one_config(&layer, x, &base, y_ref_dense);

    const float fidelity_target = 0.005f;
    rinx_result_row_t rows[256];
    int feasible = rinx_autotune_resonance(&layer, x, fidelity_target, rows, 256);
    int kept = feasible;
    if (kept > 256) kept = 256;

    // Pareto filter
    int is_pareto[256];
    for (int i = 0; i < kept; i++) is_pareto[i] = 1;
    for (int i = 0; i < kept; i++) {
        if (!is_pareto[i]) continue;
        for (int j = 0; j < kept; j++) {
            if (i == j) continue;
            if (dominates(&rows[j], &rows[i])) {
                is_pareto[i] = 0;
                break;
            }
        }
    }

    printf("| Config | Cycles/Inf | Skip%% | Refine%% | Fidelity (Norm) | Gain vs ONNX (Est) |\n");
    printf("|--------|------------|-------|---------|-----------------|-------------------|\n");
    printf("| base(x=%d,y=%d,r=%d) | %.1f | %.1f | %.1f | %.6f | %.2f |\n",
           base.x_noise_thr, base.y_noise_thr_msb, base.refine_thr_msb,
           base_row.cycles_per_inf, base_row.skip_pct, base_row.refine_pct, base_row.fidelity_norm, base_row.onnx_gain_est);

    int printed = 0;
    for (int i = 0; i < kept; i++) {
        if (!is_pareto[i]) continue;
        printf("| x=%d y=%d r=%d v=%.2g | %.1f | %.1f | %.1f | %.6f | %.2f |\n",
               rows[i].cfg.x_noise_thr,
               rows[i].cfg.y_noise_thr_msb,
               rows[i].cfg.refine_thr_msb,
               rows[i].cfg.penumbra_var_thr,
               rows[i].cycles_per_inf,
               rows[i].skip_pct,
               rows[i].refine_pct,
               rows[i].fidelity_norm,
               rows[i].onnx_gain_est);
        printed++;
        if (printed >= 12) break;
    }

    if (feasible == 0) {
        printf("\nNo feasible configs under fidelity_norm < %.6f. Consider raising refine coverage or lowering sparsity thresholds.\n", fidelity_target);
    } else {
        printf("\nFeasible configs found: %d (showing up to %d Pareto rows)\n", feasible, printed);
    }

    free_layer(&layer);
    return 0;
}
