/*
 * rin_ptsoftmax.h - Power-of-Two Softmax Implementation
 * 
 * Reemplaza: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * Con:       PTsoftmax(x_i) = 2^(x_i) >> log2_approx(sum(2^(x_j)))
 * 
 * Basado en: ICML 2025 - Sorbet: "Neuromorphic Hardware-Compatible Transformer"
 *            + Softermax (DAC 2021) + DenseShift (ICCV 2023)
 */

#ifndef RIN_PTSOFTMAX_H
#define RIN_PTSOFTMAX_H

#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURACIÓN
 * ============================================================================ */

#ifndef RIN_PTSMAX_INPUT_BITS
#define RIN_PTSMAX_INPUT_BITS  8   /* -128 a 127 */
#endif

#ifndef RIN_PTSMAX_OUTPUT_BITS
#define RIN_PTSMAX_OUTPUT_BITS 16  /* Salida Q16 */
#endif

#define RIN_PTSMAX_TABLE_SIZE 256  /* 2^8 entradas */

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_PTSoftmax_Table - Tabla LUT para 2^x en rango limitado
 * Pre-computada en inicialización
 */
typedef struct {
    uint16_t exp2_table[RIN_PTSMAX_TABLE_SIZE];  /* LUT: 2^(x/scale) */
    uint8_t  input_scale;                        /* Factor de escala del input */
    uint8_t  output_shift;                       /* Shift para normalización */
} RIN_PTSoftmax_Table;

/*
 * RIN_PTSoftmax_Config - Configuración de cuantización
 */
typedef struct {
    float   input_scale_f;      /* Escala para convertir float a índice */
    int8_t  input_zero_point;   /* Zero point para cuantización */
    uint8_t num_bits;           /* Bits de precisión (8 típicamente) */
} RIN_PTSoftmax_Config;

/* ============================================================================
 * FUNCIÓN: RIN_IntegerLog2
 * floor(log2(x)) usando clz (count leading zeros)
 * 
 * Retorna: posición del bit más significativo
 * ============================================================================ */
static inline uint32_t RIN_IntegerLog2(uint32_t x) {
    if (x == 0) return 0;
    
#if defined(__GNUC__) || defined(__clang__)
    /* 31 - __builtin_clz(x) = floor(log2(x)) para x > 0 */
    return 31 - __builtin_clz(x);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, x);
    return index;
#else
    /* Fallback manual (binary search para eficiencia) */
    uint32_t log2 = 0;
    if (x >= (1U << 16)) { log2 += 16; x >>= 16; }
    if (x >= (1U << 8))  { log2 += 8;  x >>= 8; }
    if (x >= (1U << 4))  { log2 += 4;  x >>= 4; }
    if (x >= (1U << 2))  { log2 += 2;  x >>= 2; }
    if (x >= (1U << 1))  { log2 += 1; }
    return log2;
#endif
}

/* ============================================================================
 * FUNCIÓN: RIN_NextPowerOf2
 * Redondea a siguiente potencia de 2
 * ============================================================================ */
static inline uint32_t RIN_NextPowerOf2(uint32_t x) {
    if (x <= 1) return 1;
    return 1U << (RIN_IntegerLog2(x - 1) + 1);
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_InitTable
 * Pre-computa tabla de exponenciales 2^x
 * 
 * @table:       Estructura a inicializar
 * @input_scale: Factor de escala para input (ej: 32 para x/32)
 * 
 * La tabla computa 2^(i/scale) para i en [0, 255]
 * Ajustado para inputs negativos vía indexado con offset
 * ============================================================================ */
static inline void RIN_PTSoftmax_InitTable(RIN_PTSoftmax_Table* table, uint8_t input_scale) {
    if (!table) return;
    
    table->input_scale = input_scale ? input_scale : 32;
    table->output_shift = 0;
    
    /* Precomputar 2^(x/scale) para x en rango representable */
    /* Asumimos input como signed 8-bit: -128 a 127 */
    for (int i = -128; i < 128; i++) {
        float x = (float)i / (float)table->input_scale;
        float exp2_x = powf(2.0f, x);
        
        /* Convertir a Q16 y saturar */
        uint32_t q_val = (uint32_t)(exp2_x * 65536.0f + 0.5f);
        if (q_val > 65535) q_val = 65535;
        
        /* Indexar con bias de 128 para manejar negativos */
        table->exp2_table[(uint8_t)(i + 128)] = (uint16_t)q_val;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_Lookup
 * Lookup de 2^x usando tabla
 * 
 * @table: Tabla inicializada
 * @x:     Input signed 8-bit
 * 
 * Retorna: 2^x en Q16
 * ============================================================================ */
static inline uint16_t RIN_PTSoftmax_Lookup(const RIN_PTSoftmax_Table* table, int8_t x) {
    if (!table) return 0;
    /* Indexar con bias de 128 */
    return table->exp2_table[(uint8_t)(x + 128)];
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_Compute
 * Power-of-Two Softmax completo
 * 
 * @table:  Tabla LUT inicializada
 * @input:  Array de inputs Q8 (enteros de 8 bits, signed)
 * @output: Array de outputs Q8 (probabilidades normalizadas 0-255)
 * @len:    Longitud del array
 * 
 * Algoritmo:
 *   1. Lookup 2^input[i] para cada elemento
 *   2. Sumar todos los valores
 *   3. Aproximar suma a potencia de 2: shift = log2(sum)
 *   4. Normalizar: output[i] = exp_values[i] >> shift
 * ============================================================================ */
static inline void RIN_PTSoftmax_Compute(const RIN_PTSoftmax_Table* table,
                                          const int8_t* input,
                                          uint8_t* output,
                                          uint32_t len) {
    if (!table || !input || !output || len == 0 || len > 256) return;
    
    /* Paso 1: Lookup 2^x para cada input */
    uint32_t exp_values[256];  /* Buffer temporal */
    uint64_t sum = 0;
    
    for (uint32_t i = 0; i < len; i++) {
        uint32_t exp_val = RIN_PTSoftmax_Lookup(table, input[i]);
        exp_values[i] = exp_val;
        sum += exp_val;
    }
    
    if (sum == 0) {
        /* Caso degenerado - distribución uniforme */
        uint8_t uniform_val = (uint8_t)(255 / len);
        for (uint32_t i = 0; i < len; i++) {
            output[i] = uniform_val;
        }
        return;
    }
    
    /* Paso 2: Aproximar suma a potencia de 2 más cercana */
    /* sum está en Q16, queremos log2(sum) */
    uint32_t log2_sum = RIN_IntegerLog2((uint32_t)(sum >> 16));
    if (log2_sum < 8) log2_sum = 8;  /* Mínimo para no saturar */
    
    /* Paso 3: Normalizar: output = exp_val * 255 / sum
     * sum y exp_val están en Q16 (uint32). El cociente cabe en uint8. */
    uint32_t sum32 = (uint32_t)sum;
    if (sum32 == 0) sum32 = 1;
    for (uint32_t i = 0; i < len; i++) {
        uint32_t norm = exp_values[i] * 255U / sum32;
        if (norm > 255) norm = 255;
        output[i] = (uint8_t)norm;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_ComputeOptimized
 * Versión optimizada para inferencia con buffer pre-calculado
 * 
 * @exp_values: Valores 2^x pre-calculados (uint16_t array)
 * @output:     Array de salida Q8
 * @len:        Longitud
 * 
 * Más rápido - no hace lookup, solo suma y shift
 * ============================================================================ */
static inline void RIN_PTSoftmax_ComputeOptimized(const uint16_t* exp_values,
                                                   uint8_t* output,
                                                   uint32_t len) {
    if (!exp_values || !output || len == 0) return;
    
    /* Sumar exp_values */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < len; i++) {
        sum += exp_values[i];
    }
    
    if (sum == 0) {
        memset(output, 0, len);
        return;
    }
    
    /* log2 de suma en Q16 */
    uint32_t shift = RIN_IntegerLog2((uint32_t)(sum >> 16));
    if (shift < 8) shift = 8;
    
    /* Normalizar: output = exp_val * 255 / sum */
    uint32_t sum32 = (uint32_t)sum;
    if (sum32 == 0) sum32 = 1;
    for (uint32_t i = 0; i < len; i++) {
        uint32_t norm = exp_values[i] * 255U / sum32;
        output[i] = (uint8_t)((norm > 255) ? 255 : norm);
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_Sample
 * Sampling desde distribución PTSoftmax
 * 
 * @output_probs: Array de probabilidades Q8 (0-255)
 * @len:          Longitud del array
 * @random_val:   Valor aleatorio uniforme [0, 65535]
 * 
 * Retorna: índice sampleado
 * ============================================================================ */
static inline uint32_t RIN_PTSoftmax_Sample(const uint8_t* output_probs,
                                             uint32_t len,
                                             uint16_t random_val) {
    if (!output_probs || len == 0) return 0;
    
    /* Scale random a rango acumulado */
    uint32_t cumulative = 0;
    uint32_t threshold = ((uint32_t)random_val * 255) / 65536;
    
    for (uint32_t i = 0; i < len; i++) {
        cumulative += output_probs[i];
        if (cumulative > threshold) {
            return i;
        }
    }
    
    return len - 1;  /* Fallback al último */
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_QuantizeInput
 * Cuantiza input flotante a Q8 para PTSoftmax
 * 
 * @input:      Array de floats
 * @len:        Longitud
 * @output:     Array de int8_t output
 * @config:     Configuración de cuantización
 * ============================================================================ */
static inline void RIN_PTSoftmax_QuantizeInput(const float* input,
                                              int8_t* output,
                                              uint32_t len,
                                              const RIN_PTSoftmax_Config* config) {
    if (!input || !output || !config) return;
    
    float scale = config->input_scale_f;
    int8_t zp = config->input_zero_point;
    
    for (uint32_t i = 0; i < len; i++) {
        float scaled = input[i] * scale + (float)zp;
        
        /* Saturar a int8 */
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        
        output[i] = (int8_t)(scaled + 0.5f);
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_CalibrateConfig
 * Calibra configuración de cuantización desde datos de entrenamiento
 * 
 * @training_samples: Array de arrays de floats (muestras de activación)
 * @num_samples:     Número de muestras
 * @sample_len:      Longitud de cada muestra
 * @config:          Output - configuración calibrada
 * ============================================================================ */
static inline void RIN_PTSoftmax_CalibrateConfig(const float** training_samples,
                                                  uint32_t num_samples,
                                                  uint32_t sample_len,
                                                  RIN_PTSoftmax_Config* config) {
    if (!training_samples || !config || num_samples == 0) return;
    
    /* Encontrar min/max de activaciones */
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    for (uint32_t s = 0; s < num_samples; s++) {
        const float* sample = training_samples[s];
        for (uint32_t i = 0; i < sample_len; i++) {
            if (sample[i] < min_val) min_val = sample[i];
            if (sample[i] > max_val) max_val = sample[i];
        }
    }
    
    /* Configurar escala para usar rango completo de int8 */
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1e-6f;
    
    config->input_scale_f = 255.0f / range;
    config->input_zero_point = (int8_t)(-min_val * config->input_scale_f - 128.0f);
    config->num_bits = 8;
}

/* ============================================================================
 * FUNCIÓN: RIN_PTSoftmax_ValidateAccuracy
 * Valida precisión de PTSoftmax vs softmax estándar
 * 
 * Retorna: KL-divergence promedio
 * ============================================================================ */
static inline float RIN_PTSoftmax_ValidateAccuracy(const float* reference_softmax,
                                                    const uint8_t* ptsoftmax_output,
                                                    uint32_t len) {
    if (!reference_softmax || !ptsoftmax_output || len == 0) return 1e10f;
    
    float kl_div = 0.0f;
    const float eps = 1e-10f;
    
    for (uint32_t i = 0; i < len; i++) {
        float p = reference_softmax[i];
        float q = (float)ptsoftmax_output[i] / 255.0f;
        
        if (p > eps && q > eps) {
            kl_div += p * logf(p / q);
        }
    }
    
    return kl_div;
}

/* ============================================================================
 * MACRO: RIN_PTSOFTMAX_FOR_TOKENS
 * Macro para softmax sobre vocabulario de tokens
 * ============================================================================ */
#define RIN_PTSOFTMAX_FOR_TOKENS(table, logits, vocab_size, probs) \
    RIN_PTSoftmax_Compute((table), (logits), (probs), (vocab_size))

#ifdef __cplusplus
}
#endif

#endif /* RIN_PTSOFTMAX_H */
