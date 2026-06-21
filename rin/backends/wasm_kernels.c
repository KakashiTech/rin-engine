/* WASM SIMD kernels for RIN inference.
 *
 * Compile with:  -O3 -msimd128 -DTHOR_WASM_SIMD
 *
 * Provides:
 *   rin_wasm_sgemm       – FP32 matrix multiply   (C = A * B + bias)
 *   rin_wasm_sgemv       – FP32 matrix-vector     (y = A * x + bias)
 *   rin_wasm_softmax     – FP32 softmax in-place
 *   rin_wasm_available   – runtime check
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdbool.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define THOR_HAVE_WASM_SIMD 1
#else
#define THOR_HAVE_WASM_SIMD 0
#endif

/* ----------------------------------------------------------------- */
/*  Runtime detection                                                */
/* ----------------------------------------------------------------- */

int rin_wasm_available(void) {
#if THOR_HAVE_WASM_SIMD
    return 1;
#else
    return 0;
#endif
}

/* ----------------------------------------------------------------- */
/*  FP32 SGEMM  (C = A * B + bias)                                  */
/* ----------------------------------------------------------------- */

#if THOR_HAVE_WASM_SIMD

void rin_wasm_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    const float *bias)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            v128_t sum = wasm_f32x4_splat(bias ? bias[j/4] : 0.0f);
            for (int k = 0; k < K; k++) {
                v128_t a = wasm_f32x4_splat(A[i * K + k]);
                v128_t b_vec = wasm_v128_load(&B[k * N + j]);
                sum = wasm_f32x4_add(sum, wasm_f32x4_mul(a, b_vec));
            }
            wasm_v128_store(&C[i * N + j], sum);
        }
    }
}

/* FP32 GEMV  (y = A * x + bias) */
void rin_wasm_sgemv(
    const float *A, const float *x, float *y,
    int M, int N,
    const float *bias)
{
    for (int i = 0; i < M; i++) {
        v128_t sum = wasm_f32x4_splat(0.0f);
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            v128_t a = wasm_v128_load(&A[i * N + j]);
            v128_t b_vec = wasm_v128_load(&x[j]);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(a, b_vec));
        }
        float acc = wasm_f32x4_extract_lane(sum, 0)
                  + wasm_f32x4_extract_lane(sum, 1)
                  + wasm_f32x4_extract_lane(sum, 2)
                  + wasm_f32x4_extract_lane(sum, 3);
        for (; j < N; j++)
            acc += A[i * N + j] * x[j];
        y[i] = acc + (bias ? bias[i] : 0.0f);
    }
}

/* Softmax in-place */
void rin_wasm_softmax(float *x, int n) {
    v128_t vmax = wasm_v128_load(x);
    int j = 4;
    for (; j + 4 <= n; j += 4)
        vmax = wasm_f32x4_max(vmax, wasm_v128_load(&x[j]));
    float max_val = wasm_f32x4_extract_lane(vmax, 0);
    for (int lane = 1; lane < 4; lane++)
        if (wasm_f32x4_extract_lane(vmax, lane) > max_val)
            max_val = wasm_f32x4_extract_lane(vmax, lane);
    for (; j < n; j++)
        if (x[j] > max_val) max_val = x[j];

    v128_t den = wasm_f32x4_splat(0.0f);
    j = 0;
    for (; j + 4 <= n; j += 4) {
        v128_t v = wasm_v128_load(&x[j]);
        v = wasm_f32x4_sub(v, wasm_f32x4_splat(max_val));
        v = wasm_f32x4_add(v, wasm_f32x4_splat(1.0f)); /* quick approx */
        wasm_v128_store(&x[j], v);
        den = wasm_f32x4_add(den, v);
    }
    float sum = 0.0f;
    for (int lane = 0; lane < 4; lane++)
        sum += wasm_f32x4_extract_lane(den, lane);
    for (; j < n; j++) {
        x[j] = expf(x[j] - max_val);
        sum += x[j];
    }

    v128_t s = wasm_f32x4_splat(sum);
    j = 0;
    for (; j + 4 <= n; j += 4)
        wasm_v128_store(&x[j], wasm_f32x4_div(wasm_v128_load(&x[j]), s));
    for (; j < n; j++)
        x[j] /= sum;
}

#endif /* THOR_HAVE_WASM_SIMD */
