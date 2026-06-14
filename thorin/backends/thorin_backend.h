/* THOR Backend Kernel API — unified interface for NEON / WASM / x86 kernels.
 *
 * Each backend provides drop-in replacements for matmul, gemv, and softmax.
 * If a backend is not compiled in, its functions are no-ops returning 0.
 */

#ifndef THOR_BACKEND_H
#define THOR_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------- */
/*  ARM NEON                                                         */
/* ----------------------------------------------------------------- */

int  thor_neon_available(void);
void thor_neon_sgemm(const float *A, const float *B, float *C,
                     int M, int N, int K, const float *bias);
void thor_neon_sgemv(const float *A, const float *x, float *y,
                     int M, int N, const float *bias);
void thor_neon_softmax(float *x, int n);
void thor_neon_int8_gemm(const int8_t *A, const int8_t *B, int32_t *C,
                         int M, int N, int K, const int32_t *bias);

/* ----------------------------------------------------------------- */
/*  WASM SIMD                                                        */
/* ----------------------------------------------------------------- */

int  thor_wasm_available(void);
void thor_wasm_sgemm(const float *A, const float *B, float *C,
                     int M, int N, int K, const float *bias);
void thor_wasm_sgemv(const float *A, const float *x, float *y,
                     int M, int N, const float *bias);
void thor_wasm_softmax(float *x, int n);

/* ----------------------------------------------------------------- */
/*  Auto-detection helper                                            */
/* ----------------------------------------------------------------- */

/* Returns a human-readable string like "neon", "wasm", "avx2", "generic". */
const char* thor_best_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* THOR_BACKEND_H */
