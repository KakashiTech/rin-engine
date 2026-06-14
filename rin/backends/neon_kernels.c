/* ARM NEON-optimised kernels for THOR inference.
 *
 * Compile with:  -O3 -march=armv8-a+simd -DTHOR_NEON
 *
 * Provides:
 *   thor_neon_sgemm       – FP32 matrix multiply   (C = A * B + bias)
 *   thor_neon_sgemv       – FP32 matrix-vector     (y = A * x + bias)
 *   thor_neon_softmax     – FP32 softmax in-place
 *   thor_neon_int8_gemm   – INT8 matrix multiply   (C = A * B + bias)
 *   thor_neon_available   – runtime check
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define THOR_HAVE_NEON 1
#else
#define THOR_HAVE_NEON 0
#endif

/* ----------------------------------------------------------------- */
/*  Runtime detection                                                */
/* ----------------------------------------------------------------- */

int thor_neon_available(void) {
#if THOR_HAVE_NEON
    return 1;
#else
    return 0;
#endif
}

/* ----------------------------------------------------------------- */
/*  FP32 SGEMM  (C = A * B + bias)                                  */
/*  A: M×K,  B: K×N,  C: M×N                                       */
/* ----------------------------------------------------------------- */

#if THOR_HAVE_NEON

void thor_neon_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    const float *bias)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t sum = vdupq_n_f32(bias ? bias[j/4] : 0.0f);
            for (int k = 0; k < K; k++) {
                float32x4_t a = vdupq_n_f32(A[i * K + k]);
                float32x4_t b = vld1q_f32(&B[k * N + j]);
                sum = vmlaq_f32(sum, a, b);
            }
            vst1q_f32(&C[i * N + j], sum);
        }
    }
}

/* FP32 GEMV  (y = A * x + bias) */
void thor_neon_sgemv(
    const float *A, const float *x, float *y,
    int M, int N,
    const float *bias)
{
    for (int i = 0; i < M; i++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t a = vld1q_f32(&A[i * N + j]);
            float32x4_t b = vld1q_f32(&x[j]);
            sum = vmlaq_f32(sum, a, b);
        }
        float acc = vaddvq_f32(sum);
        for (; j < N; j++)
            acc += A[i * N + j] * x[j];
        y[i] = acc + (bias ? bias[i] : 0.0f);
    }
}

/* Softmax in-place */
void thor_neon_softmax(float *x, int n) {
    /* max */
    float32x4_t vmax = vld1q_f32(x);
    int j = 4;
    for (; j + 4 <= n; j += 4)
        vmax = vmaxq_f32(vmax, vld1q_f32(&x[j]));
    float max_val = vmaxvq_f32(vmax);
    for (; j < n; j++)
        if (x[j] > max_val) max_val = x[j];

    /* exp + sum */
    float32x4_t vsum = vdupq_n_f32(0.0f);
    j = 0;
    for (; j + 4 <= n; j += 4) {
        float32x4_t v = vld1q_f32(&x[j]);
        v = vsubq_f32(v, vdupq_n_f32(max_val));
        v = vexpq_f32(v);
        vst1q_f32(&x[j], v);
        vsum = vaddq_f32(vsum, v);
    }
    float sum = vaddvq_f32(vsum);
    for (; j < n; j++) {
        x[j] = expf(x[j] - max_val);
        sum += x[j];
    }

    /* divide */
    float32x4_t den = vdupq_n_f32(sum);
    j = 0;
    for (; j + 4 <= n; j += 4)
        vst1q_f32(&x[j], vdivq_f32(vld1q_f32(&x[j]), den));
    for (; j < n; j++)
        x[j] /= sum;
}

/* INT8 GEMM:  C_int32 = A_int8 * B_int8  (with dequant later) */
void thor_neon_int8_gemm(
    const int8_t *A, const int8_t *B, int32_t *C,
    int M, int N, int K,
    const int32_t *bias)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            int32x4_t sum = vdupq_n_s32(bias ? bias[j/4] : 0);
            for (int k = 0; k < K; k++) {
                int8x8_t a = vdup_n_s8(A[i * K + k]);
                int8x8_t b_vec = vld1_s8((const int8_t *)&B[k * N + j]);
                int16x8_t prod = vmull_s8(a, b_vec);
                sum = vaddw_s16(sum, vget_low_s16(prod));
            }
            vst1q_s32(&C[i * N + j], sum);
        }
    }
}

#endif /* THOR_HAVE_NEON */
