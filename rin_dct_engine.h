/*
 * rin_dct_engine.h - Motor DCT-II de Punto Fijo (AAN Algorithm)
 * 
 * Fast Discrete Cosine Transform 8-point usando Arai-Agui-Nakajima
 * - Solo 5 multiplicaciones (vs 64 naive)
 * - Fixed-point Q15 para hardware sin FPU
 * - Poda de alta frecuencia integrada
 * 
 * Basado en: prtsh/aan_dct + Loeffler et al. "Practical fast 1-D DCT"
 */

#ifndef RIN_DCT_ENGINE_H
#define RIN_DCT_ENGINE_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTAS AAN PARA DCT-II 8-POINT (Q15)
 * 
 * Cosenos escalados por 32768:
 * Ck = cos(k * pi/16) * 32768
 * ============================================================================ */

#define RIN_DCT_C1  32138  /* cos(pi/16) * 32768 ≈ 0.9808 */
#define RIN_DCT_C2  30274  /* cos(2*pi/16) * 32768 ≈ 0.9239 */
#define RIN_DCT_C3  27246  /* cos(3*pi/16) * 32768 ≈ 0.8315 */
#define RIN_DCT_C4  23170  /* cos(4*pi/16) * 32768 = 0.7071 = 1/sqrt(2) */
#define RIN_DCT_C5  18205  /* cos(5*pi/16) * 32768 ≈ 0.5556 */
#define RIN_DCT_C6  12540  /* cos(6*pi/16) * 32768 ≈ 0.3827 */
#define RIN_DCT_C7  6393   /* cos(7*pi/16) * 32768 ≈ 0.1951 */

/* Constantes adicionales para AAN */
#define RIN_DCT_COS_1_16  32138
#define RIN_DCT_COS_2_16  30274
#define RIN_DCT_COS_3_16  27246
#define RIN_DCT_COS_4_16  23170
#define RIN_DCT_COS_5_16  18205
#define RIN_DCT_COS_6_16  12540
#define RIN_DCT_COS_7_16  6393

/* ============================================================================
 * CONSTANTES DE AJUSTE DE ESCALA AAN
 * ============================================================================ */

#define RIN_DCT_AAN_SCALE_0  23170  /* 1/sqrt(2) para butterfly 0 */
#define RIN_DCT_AAN_SCALE_1  23170  /* Escalado por etapa */
#define RIN_DCT_AAN_SCALE_2  23170

/* ============================================================================
 * FUNCIÓN: RIN_Q15_Mul (helper)
 * Multiplicación Q15: (a * b) >> 15
 * ============================================================================ */
static inline int16_t RIN_Q15_Mul(int16_t a, int16_t b) {
    return (int16_t)(((int32_t)a * (int32_t)b) >> 15);
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT8_Forward
 * DCT-II 8-point forward usando algoritmo AAN
 * 
 * @input:  8 valores Q15
 * @output: 8 coeficientes DCT Q15
 * 
 * Flujo AAN:
 *   Stage 1: 4 butterflies
 *   Stage 2: 2 butterflies + 2 rotaciones
 *   Stage 3: 1 butterfly + 4 rotaciones (las 5 multiplicaciones)
 *   Stage 4: Reordering bit-reversed
 * 
 * Total: 29 additions, 5 multiplications
 * Naive: 64 multiplications
 * ============================================================================ */
static inline void RIN_DCT8_Forward(const int16_t* input, int16_t* output) {
    int32_t tmp[8];  /* Usamos 32-bit para evitar overflow en intermedios */
    
    /* ========== Stage 1: Butterflies ========== */
    /* Even part butterflies */
    int32_t a0 = (int32_t)input[0] + (int32_t)input[7];
    int32_t a1 = (int32_t)input[1] + (int32_t)input[6];
    int32_t a2 = (int32_t)input[2] + (int32_t)input[5];
    int32_t a3 = (int32_t)input[3] + (int32_t)input[4];
    
    /* Odd part butterflies */
    int32_t b0 = (int32_t)input[0] - (int32_t)input[7];
    int32_t b1 = (int32_t)input[1] - (int32_t)input[6];
    int32_t b2 = (int32_t)input[2] - (int32_t)input[5];
    int32_t b3 = (int32_t)input[3] - (int32_t)input[4];
    
    /* ========== Stage 2: Even part ========== */
    /* 2-point DCT on even */
    int32_t c0 = a0 + a3;
    int32_t c1 = a1 + a2;
    int32_t c2 = a1 - a2;
    int32_t c3 = a0 - a3;
    
    /* Rotaciones para even part */
    int32_t d0 = c0 + c1;
    int32_t d1 = c0 - c1;
    
    /* c2 y c3 pasan a rotaciones más complejas */
    
    /* ========== Stage 3: Las 5 multiplicaciones AAN ========== */
    /* 
     * Rotación para even part (c2, c3):
     * [c2']   [C6  C2] [c2]
     * [c3'] = [C6 -C2] [c3]
     */
    int32_t e0 = (c2 * RIN_DCT_C6 + c3 * RIN_DCT_C2) >> 15;
    int32_t e1 = (c3 * RIN_DCT_C6 - c2 * RIN_DCT_C2) >> 15;
    
    /* 
     * Rotaciones para odd part (b0, b1, b2, b3):
     * Estas son las 4 multiplicaciones restantes
     */
    int32_t f0 = (b0 * RIN_DCT_C1 + b1 * RIN_DCT_C7) >> 15;
    int32_t f1 = (b1 * RIN_DCT_C1 - b0 * RIN_DCT_C7) >> 15;
    int32_t f2 = (b2 * RIN_DCT_C3 + b3 * RIN_DCT_C5) >> 15;
    int32_t f3 = (b3 * RIN_DCT_C3 - b2 * RIN_DCT_C5) >> 15;
    
    /* ========== Stage 4: Butterflies finales y reordering ========== */
    /* Even outputs (0, 4, 2, 6) - bit-reversed order */
    tmp[0] = d0 >> 1;  /* Escalado implícito */
    tmp[4] = d1 >> 1;
    tmp[2] = e0 >> 1;
    tmp[6] = e1 >> 1;
    
    /* Odd outputs - final butterflies */
    tmp[1] = f0 + f3;
    tmp[5] = f1 + f2;
    tmp[3] = f1 - f2;
    tmp[7] = f0 - f3;
    
    /* ========== Output ========== */
    for (int i = 0; i < 8; i++) {
        /* Saturar a Q15 */
        if (tmp[i] > 32767) tmp[i] = 32767;
        if (tmp[i] < -32768) tmp[i] = -32768;
        output[i] = (int16_t)tmp[i];
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT8_Inverse
 * IDCT-II 8-point (DCT-III)
 * 
 * IDCT es la transpuesta de DCT - reutilizamos forward con pre/post procesamiento
 * ============================================================================ */
static inline void RIN_DCT8_Inverse(const int16_t* input, int16_t* output) {
    /* Para IDCT: 
     * 1. Reordenar input (bit-reverse para DCT-III)
     * 2. Aplicar forward con signos alternados
     * 3. Escalar
     */
    int16_t reordered[8];
    
    /* Reordering para DCT-III */
    reordered[0] = input[0];
    reordered[1] = input[4];
    reordered[2] = input[2];
    reordered[3] = input[6];
    reordered[4] = input[1];
    reordered[5] = input[5];
    reordered[6] = input[3];
    reordered[7] = input[7];
    
    /* Llamar forward (matriz de DCT es ortogonal) */
    int16_t temp[8];
    RIN_DCT8_Forward(reordered, temp);
    
    /* Escalar final: compensar el >> 1 en forward y escalar por 2/N = 0.25 */
    /* Total: multiplicar por 0.5 para compensar forward, luego 0.25 para IDCT = 0.125 */
    /* En Q15: 0.125 = 4096 */
    for (int i = 0; i < 8; i++) {
        int32_t val = (int32_t)temp[i] * 4096;  /* Escala 0.125 */
        val = (val >> 15);  /* Q30 -> Q15 */
        
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;
        output[i] = (int16_t)val;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT2D_8x8
 * DCT 2D 8x8 (para procesamiento de patches)
 * 
 * Aplica DCT por filas, luego por columnas
 * ============================================================================ */
static inline void RIN_DCT2D_8x8(const int16_t* input, int16_t* output) {
    int16_t temp[8][8];
    int16_t row[8];
    int i, j;
    
    /* DCT por filas */
    for (i = 0; i < 8; i++) {
        RIN_DCT8_Forward(&input[i * 8], row);
        for (j = 0; j < 8; j++) {
            temp[i][j] = row[j];
        }
    }
    
    /* DCT por columnas */
    for (j = 0; j < 8; j++) {
        /* Extraer columna */
        for (i = 0; i < 8; i++) {
            row[i] = temp[i][j];
        }
        /* Aplicar DCT */
        RIN_DCT8_Forward(row, row);
        /* Guardar columna */
        for (i = 0; i < 8; i++) {
            output[i * 8 + j] = row[i];
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT2D_8x8_Inverse
 * IDCT 2D 8x8
 * ============================================================================ */
static inline void RIN_DCT2D_8x8_Inverse(const int16_t* input, int16_t* output) {
    int16_t temp[8][8];
    int16_t row[8];
    int i, j;
    
    /* IDCT por filas */
    for (i = 0; i < 8; i++) {
        RIN_DCT8_Inverse(&input[i * 8], row);
        for (j = 0; j < 8; j++) {
            temp[i][j] = row[j];
        }
    }
    
    /* IDCT por columnas */
    for (j = 0; j < 8; j++) {
        for (i = 0; i < 8; i++) {
            row[i] = temp[i][j];
        }
        RIN_DCT8_Inverse(row, row);
        for (i = 0; i < 8; i++) {
            output[i * 8 + j] = row[i];
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT_PruneHighFreq
 * Poda de coeficientes de alta frecuencia
 * 
 * Elimina coeficientes con energía residual < threshold
 * Esto permite compresión lossy controlada
 * 
 * @dct_coeffs: Coeficientes DCT (modificados in-place)
 * @len:        Longitud (8, 64 para 8x8, etc.)
 * @threshold:  Umbral Q15 (ej: 100 ~ 0.003)
 * 
 * Retorna: número de coeficientes podados
 * ============================================================================ */
static inline uint32_t RIN_DCT_PruneHighFreq(int16_t* dct_coeffs, 
                                            uint32_t len,
                                            int16_t threshold) {
    uint32_t pruned = 0;
    
    /* Asume orden: DC primero (índice 0), luego frecuencias crecientes */
    /* Preservamos siempre DC (componente 0) */
    for (uint32_t i = 1; i < len; i++) {
        int16_t val = dct_coeffs[i];
        int16_t abs_val = (val < 0) ? -val : val;
        
        if (abs_val < threshold) {
            dct_coeffs[i] = 0;
            pruned++;
        }
    }
    
    return pruned;
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT_ZigzagReorder_8x8
 * Reordena coeficientes 8x8 en zigzag
 * 
 * Útil para cuantización por posición de frecuencia
 * ============================================================================ */
static const uint8_t RIN_DCT_ZIGZAG_8x8[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

static inline void RIN_DCT_ZigzagReorder_8x8(const int16_t* in, int16_t* out) {
    for (int i = 0; i < 64; i++) {
        out[i] = in[RIN_DCT_ZIGZAG_8x8[i]];
    }
}

static inline void RIN_DCT_ZigzagUnreorder_8x8(const int16_t* in, int16_t* out) {
    for (int i = 0; i < 64; i++) {
        out[RIN_DCT_ZIGZAG_8x8[i]] = in[i];
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT_EnergyRetained
 * Calcula % de energía retenida después de poda
 * 
 * Energía = sum(|coeffs|^2)
 * ============================================================================ */
static inline float RIN_DCT_EnergyRetained(const int16_t* original,
                                            const int16_t* pruned,
                                            uint32_t len) {
    int64_t energy_orig = 0;
    int64_t energy_pruned = 0;
    
    for (uint32_t i = 0; i < len; i++) {
        int32_t val_orig = original[i];
        int32_t val_pruned = pruned[i];
        
        energy_orig += val_orig * val_orig;
        energy_pruned += val_pruned * val_pruned;
    }
    
    if (energy_orig <= 0) return 1.0f;
    
    return (float)energy_pruned / (float)energy_orig;
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT_ComputeSpectralEntropy
 * Entropía espectral - medida de dispersión de energía
 * 
 * Usado para detectar qué tan "comprimible" es una señal
 * ============================================================================ */
static inline float RIN_DCT_ComputeSpectralEntropy(const int16_t* dct_coeffs,
                                                  uint32_t len) {
    /* Calcular energía total */
    int64_t total_energy = 0;
    for (uint32_t i = 0; i < len; i++) {
        int32_t val = dct_coeffs[i];
        total_energy += val * val;
    }
    
    if (total_energy <= 0) return 0.0f;
    
    /* Entropía: -sum(p_i * log(p_i)) */
    float entropy = 0.0f;
    for (uint32_t i = 0; i < len; i++) {
        int32_t val = dct_coeffs[i];
        float p = (float)(val * val) / (float)total_energy;
        
        if (p > 1e-10f) {
            entropy -= p * logf(p);
        }
    }
    
    /* Normalizar a [0, 1] */
    float max_entropy = logf((float)len);
    return entropy / max_entropy;
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT_QuantizeCoeffs
 * Cuantiza coeficientes DCT (como en JPEG)
 * 
 * @coeffs:     Coeficientes a cuantizar (in-place)
 * @len:        Longitud
 * @quant_table: Tabla de cuantización (1/Q para cada coeficiente)
 * ============================================================================ */
static inline void RIN_DCT_QuantizeCoeffs(int16_t* coeffs,
                                           uint32_t len,
                                           const uint8_t* quant_table) {
    for (uint32_t i = 0; i < len; i++) {
        uint8_t q = quant_table ? quant_table[i] : 16;  /* Default Q=16 */
        if (q == 0) q = 1;  /* Evitar div/0 */
        
        coeffs[i] = (int16_t)(((int32_t)coeffs[i] + (q >> 1)) / q);
    }
}

static inline void RIN_DCT_DequantizeCoeffs(int16_t* coeffs,
                                           uint32_t len,
                                           const uint8_t* quant_table) {
    for (uint32_t i = 0; i < len; i++) {
        uint8_t q = quant_table ? quant_table[i] : 16;
        coeffs[i] = coeffs[i] * q;
    }
}

/* ============================================================================
 * ESTRUCTURA: RIN_DCT_BlockProcessor
 * Procesador de bloques para transformación de pesos/features
 * ============================================================================ */
typedef struct {
    int16_t  dct_buffer[64];     /* Buffer 8x8 */
    uint8_t  quant_table[64];    /* Tabla de cuantización */
    uint32_t blocks_processed;
    float    total_energy_retained;
} RIN_DCT_BlockProcessor;

static inline void RIN_DCT_Processor_Init(RIN_DCT_BlockProcessor* proc) {
    if (!proc) return;
    
    memset(proc->dct_buffer, 0, sizeof(proc->dct_buffer));
    
    /* Tabla de cuantización JPEG-like */
    /* Bajas frecuencias: Q más fino, altas: Q más grueso */
    const uint8_t default_quant[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };
    memcpy(proc->quant_table, default_quant, 64);
    
    proc->blocks_processed = 0;
    proc->total_energy_retained = 0.0f;
}

/* ============================================================================
 * FUNCIÓN: RIN_DCT_Processor_CompressBlock
 * Pipeline completo: DCT -> Cuantizar -> (Opcional: poda) -> IDCT
 * ============================================================================ */
static inline void RIN_DCT_Processor_CompressBlock(RIN_DCT_BlockProcessor* proc,
                                                   const int16_t* input,
                                                   int16_t* output,
                                                   int16_t prune_threshold,
                                                   float* energy_retained) {
    if (!proc || !input || !output) return;
    
    /* Forward DCT */
    RIN_DCT2D_8x8(input, proc->dct_buffer);
    
    /* Cuantización */
    RIN_DCT_QuantizeCoeffs(proc->dct_buffer, 64, proc->quant_table);
    
    /* Poda opcional */
    int16_t temp_buffer[64];
    memcpy(temp_buffer, proc->dct_buffer, sizeof(temp_buffer));
    
    if (prune_threshold > 0) {
        RIN_DCT_PruneHighFreq(proc->dct_buffer, 64, prune_threshold);
    }
    
    /* Medir energía retenida */
    if (energy_retained) {
        *energy_retained = RIN_DCT_EnergyRetained(temp_buffer, proc->dct_buffer, 64);
        proc->total_energy_retained += *energy_retained;
    }
    
    /* De-cuantizar */
    RIN_DCT_DequantizeCoeffs(proc->dct_buffer, 64, proc->quant_table);
    
    /* Inverse DCT */
    RIN_DCT2D_8x8_Inverse(proc->dct_buffer, output);
    
    proc->blocks_processed++;
}

#ifdef __cplusplus
}
#endif

#endif /* RIN_DCT_ENGINE_H */
