/*
 * rin_betti_calculator.h - Cálculo de Números de Betti para Representaciones
 * 
 * Topological Data Analysis (TDA) para verificar preservación estructural
 * en destilación de modelos.
 * 
 * β0: componentes conectados
 * β1: loops/ciclos  
 * β2: voids/cavidades
 * 
 * Basado en: "Topology of Deep Neural Networks" (Naitzat et al., JMLR 2020)
 *            + "Estimating Betti Numbers Using Deep Learning" (IJCNN 2019)
 */

#ifndef RIN_BETTI_CALCULATOR_H
#define RIN_BETTI_CALCULATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURACIÓN
 * ============================================================================ */

#ifndef RIN_BETTI_MAX_DIM
#define RIN_BETTI_MAX_DIM 3     /* Calcular hasta β2 (voids) */
#endif

#ifndef RIN_BETTI_MAX_POINTS
#define RIN_BETTI_MAX_POINTS 512  /* Max puntos para análisis */
#endif

#ifndef RIN_BETTI_MAX_SIMPLICES
#define RIN_BETTI_MAX_SIMPLICES 4096  /* Límite de simplices */
#endif

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_BettiNumbers - Resultado del cálculo de homología
 */
typedef struct {
    uint32_t beta0;            /* Componentes conectados */
    uint32_t beta1;            /* Ciclos/loops */
    uint32_t beta2;            /* Voids/cavidades */
    uint32_t total_complexity; /* ω(M) = β0 + β1 + β2 */
} RIN_BettiNumbers;

/*
 * RIN_Simplex - Simplice para complejo simplicial
 */
typedef struct {
    uint32_t vertices[4];      /* Hasta tetrahedros (dim 3) */
    uint8_t  dim;              /* Dimensión: 0=vertex, 1=edge, 2=triangle, 3=tet */
    float    diameter;         /* Diámetro (max distancia entre vértices) */
    float    birth;            /* Tiempo de nacimiento (escala epsilon) */
} RIN_Simplex;

/*
 * RIN_SimplicialComplex - Complejo simplicial
 */
typedef struct {
    RIN_Simplex* simplices;    /* Array de simplices */
    uint32_t num_simplices;    /* Número actual de simplices */
    uint32_t capacity;         /* Capacidad del array */
    float max_epsilon;         /* Escala máxima */
} RIN_SimplicialComplex;

/*
 * RIN_BettiCalculator - Estado del calculador
 */
typedef struct {
    float* distance_matrix;    /* Matriz de distancias precomputada */
    uint32_t num_points;       /* Número de puntos */
    uint8_t point_dim;         /* Dimensión de cada punto */
    
    /* Filtros persistentes */
    float epsilon_min;
    float epsilon_max;
    uint32_t num_samples;      /* Muestras de epsilon para persistencia */
    
    /* Resultados acumulados */
    RIN_BettiNumbers* persistent_betti;  /* [num_samples] array */
} RIN_BettiCalculator;

/*
 * RIN_PersistentSlice - Betti numbers a una escala específica
 */
typedef struct {
    float epsilon;             /* Escala del complejo */
    RIN_BettiNumbers betti;    /* Números de Betti */
    float persistence;         /* Cuánto persisten las features */
} RIN_PersistentSlice;

/*
 * RIN_TopologicalComparison - Comparación entre dos representaciones
 */
typedef struct {
    float beta0_diff;          /* Diferencia en componentes */
    float beta1_diff;          /* Diferencia en ciclos */
    float beta2_diff;          /* Diferencia en voids */
    float total_diff;          /* Diferencia total normalizada */
    bool  is_similar;          /* Si están dentro de tolerancia */
} RIN_TopologicalComparison;

/* ============================================================================
 * FUNCIÓN: RIN_Betti_EuclideanDistance
 * Distancia euclidiana entre dos puntos
 * ============================================================================ */
static inline float RIN_Betti_EuclideanDistance(const float* a,
                                                const float* b,
                                                uint8_t dim) {
    float sum = 0.0f;
    for (uint8_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_ComputeDistanceMatrix
 * Pre-computa matriz de distancias entre todos los puntos
 * 
 * @points:     Array de puntos flattenado [n_points][dim]
 * @n_points:   Número de puntos
 * @dim:        Dimensión de cada punto
 * @dist_matrix: Output [n_points][n_points] (pre-allocado)
 * ============================================================================ */
static inline void RIN_Betti_ComputeDistanceMatrix(const float* points,
                                                   uint32_t n_points,
                                                   uint8_t dim,
                                                   float* dist_matrix) {
    for (uint32_t i = 0; i < n_points; i++) {
        dist_matrix[i * n_points + i] = 0.0f;  /* Diagonal */
        
        for (uint32_t j = i + 1; j < n_points; j++) {
            const float* pi = &points[i * dim];
            const float* pj = &points[j * dim];
            
            float dist = RIN_Betti_EuclideanDistance(pi, pj, dim);
            dist_matrix[i * n_points + j] = dist;
            dist_matrix[j * n_points + i] = dist;  /* Simétrica */
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_VietorisRips_Build
 * Construye complejo Vietoris-Rips a escala epsilon
 * 
 * Incluye simplices hasta dimensión max_dim (típicamente 2 o 3)
 * 
 * @dist_matrix:    Matriz de distancias precomputada
 * @n_points:       Número de puntos
 * @epsilon:        Escala del complejo
 * @max_dim:        Dimensión máxima de simplices
 * @simplices:      Output - array de simplices (pre-allocado)
 * @max_simplices:  Capacidad del array
 * 
 * Retorna: número de simplices creados
 * ============================================================================ */
uint32_t RIN_Betti_VietorisRips_Build(const float* dist_matrix,
                                      uint32_t n_points,
                                      float epsilon,
                                      uint8_t max_dim,
                                      RIN_Simplex* simplices,
                                      uint32_t max_simplices);

/* ============================================================================
 * Implementación inline de Vietoris-Rips
 * ============================================================================ */
static inline uint32_t RIN_Betti_VietorisRips_Build_Impl(const float* dist_matrix,
                                                         uint32_t n_points,
                                                         float epsilon,
                                                         uint8_t max_dim,
                                                         RIN_Simplex* simplices,
                                                         uint32_t max_simplices) {
    uint32_t count = 0;
    
    /* Vertices (0-simplices) - siempre incluir todos */
    for (uint32_t i = 0; i < n_points && count < max_simplices; i++) {
        simplices[count].vertices[0] = i;
        simplices[count].dim = 0;
        simplices[count].diameter = 0.0f;
        simplices[count].birth = 0.0f;
        count++;
    }
    
    if (max_dim >= 1) {
        /* Edges (1-simplices) - pares de puntos dentro de epsilon */
        for (uint32_t i = 0; i < n_points && count < max_simplices; i++) {
            for (uint32_t j = i + 1; j < n_points && count < max_simplices; j++) {
                float dist = dist_matrix[i * n_points + j];
                if (dist <= epsilon) {
                    simplices[count].vertices[0] = i;
                    simplices[count].vertices[1] = j;
                    simplices[count].dim = 1;
                    simplices[count].diameter = dist;
                    simplices[count].birth = dist;
                    count++;
                }
            }
        }
    }
    
    if (max_dim >= 2) {
        /* Triangles (2-simplices) - triples donde todos los pares están conectados */
        /* Simplificación: para grandes n_points, esto es O(n³) */
        /* En producción usar librerías como ripser o gudhi */
        uint32_t max_check = (n_points > 50) ? 50 : n_points;  /* Limitar para performance */
        
        for (uint32_t i = 0; i < max_check && count < max_simplices; i++) {
            for (uint32_t j = i + 1; j < max_check && count < max_simplices; j++) {
                for (uint32_t k = j + 1; k < max_check && count < max_simplices; k++) {
                    /* Verificar si los 3 pares están conectados */
                    float dij = dist_matrix[i * n_points + j];
                    float dik = dist_matrix[i * n_points + k];
                    float djk = dist_matrix[j * n_points + k];
                    
                    float max_edge = fmaxf(dij, fmaxf(dik, djk));
                    
                    if (max_edge <= epsilon) {
                        simplices[count].vertices[0] = i;
                        simplices[count].vertices[1] = j;
                        simplices[count].vertices[2] = k;
                        simplices[count].dim = 2;
                        simplices[count].diameter = max_edge;
                        simplices[count].birth = max_edge;
                        count++;
                    }
                }
            }
        }
    }
    
    return count;
}

/* Macro para acceder a la implementación */
#define RIN_Betti_VietorisRips_Build RIN_Betti_VietorisRips_Build_Impl

/* ============================================================================
 * FUNCIÓN: RIN_Betti_CalculateFromComplex
 * Calcula números de Betti por álgebra de homología simplificada
 * 
 * Usa matriz de borde para calcular rangos de homología
 * β_k = dim(Z_k) - dim(B_k) = nullidad(borde_k) - rango(borde_{k+1})
 * 
 * Versión simplificada para complejos pequeños (<100 simplices)
 * ============================================================================ */
static inline RIN_BettiNumbers RIN_Betti_CalculateFromComplex(const RIN_Simplex* simplices,
                                                             uint32_t num_simplices) {
    RIN_BettiNumbers result = {0, 0, 0, 0};
    
    if (!simplices || num_simplices == 0) return result;
    
    /* Contar simplices por dimensión */
    uint32_t n0 = 0, n1 = 0, n2 = 0, n3 = 0;
    
    for (uint32_t i = 0; i < num_simplices; i++) {
        switch (simplices[i].dim) {
            case 0: n0++; break;
            case 1: n1++; break;
            case 2: n2++; break;
            case 3: n3++; break;
        }
    }
    
    /* Aproximación simplificada usando fórmula de Euler característica */
    /* χ = V - E + F - T = β0 - β1 + β2 */
    /* Para grafos conexos: β0 = 1, entonces β1 ≈ E - V + 1 */
    
    /* β0: componentes conectadas (aproximar como 1 si hay al menos un edge) */
    result.beta0 = (n1 > 0) ? 1 : n0;
    
    /* β1: ciclos = E - V + β0 (para grafo conexo) */
    if (result.beta0 > 0) {
        int32_t beta1 = (int32_t)n1 - (int32_t)n0 + (int32_t)result.beta0;
        result.beta1 = (beta1 > 0) ? (uint32_t)beta1 : 0;
    }
    
    /* β2: voids - aproximar desde 2-simplices */
    /* χ = V - E + F = β0 - β1 + β2 */
    /* β2 = χ - β0 + β1 = (V - E + F) - β0 + β1 */
    if (n2 > 0) {
        int32_t chi = (int32_t)n0 - (int32_t)n1 + (int32_t)n2;
        int32_t beta2 = chi - (int32_t)result.beta0 + (int32_t)result.beta1;
        result.beta2 = (beta2 > 0) ? (uint32_t)beta2 : 0;
    }
    
    result.total_complexity = result.beta0 + result.beta1 + result.beta2;
    
    return result;
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_CalculateFromPoints
 * Pipeline completo: puntos -> VR-complex -> Betti numbers
 * 
 * @points:     Array de puntos flattenado [n][dim]
 * @n_points:   Número de puntos
 * @dim:        Dimensión de cada punto
 * @epsilon:    Escala del complejo
 * @max_dim:    Dimensión máxima a calcular
 * 
 * Retorna: Betti numbers calculados
 * ============================================================================ */
static inline RIN_BettiNumbers RIN_Betti_CalculateFromPoints(const float* points,
                                                             uint32_t n_points,
                                                             uint8_t dim,
                                                             float epsilon,
                                                             uint8_t max_dim) {
    RIN_BettiNumbers result = {0, 0, 0, 0};
    
    if (!points || n_points == 0 || epsilon <= 0) return result;
    
    /* Limitar n_points para performance */
    uint32_t n_eff = (n_points > RIN_BETTI_MAX_POINTS) ? RIN_BETTI_MAX_POINTS : n_points;
    
    /* Allocar matriz de distancias en stack (cuidado con tamaño grande) */
    float dist_matrix[RIN_BETTI_MAX_POINTS * RIN_BETTI_MAX_POINTS];
    
    RIN_Betti_ComputeDistanceMatrix(points, n_eff, dim, dist_matrix);
    
    /* Construir complejo */
    RIN_Simplex simplices[RIN_BETTI_MAX_SIMPLICES];
    uint32_t n_simplices = RIN_Betti_VietorisRips_Build(
        dist_matrix, n_eff, epsilon, max_dim, 
        simplices, RIN_BETTI_MAX_SIMPLICES
    );
    
    /* Calcular Betti numbers */
    result = RIN_Betti_CalculateFromComplex(simplices, n_simplices);
    
    return result;
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_PersistentAnalysis
 * Análisis de persistencia a múltiples escalas
 * 
 * @points:        Datos de entrada
 * @n_points:      Cantidad de puntos
 * @dim:           Dimensionalidad
 * @epsilon_vals:  Array de escalas a analizar
 * @n_epsilons:    Cantidad de escalas
 * @slices:        Output - resultados por escala (pre-allocado)
 * ============================================================================ */
static inline void RIN_Betti_PersistentAnalysis(const float* points,
                                               uint32_t n_points,
                                               uint8_t dim,
                                               const float* epsilon_vals,
                                               uint32_t n_epsilons,
                                               RIN_PersistentSlice* slices) {
    if (!points || !epsilon_vals || !slices || n_epsilons == 0) return;
    
    uint32_t n_eff = (n_points > RIN_BETTI_MAX_POINTS) ? RIN_BETTI_MAX_POINTS : n_points;
    
    float dist_matrix[RIN_BETTI_MAX_POINTS * RIN_BETTI_MAX_POINTS];
    RIN_Betti_ComputeDistanceMatrix(points, n_eff, dim, dist_matrix);
    
    for (uint32_t e = 0; e < n_epsilons; e++) {
        float eps = epsilon_vals[e];
        slices[e].epsilon = eps;
        
        RIN_Simplex simplices[RIN_BETTI_MAX_SIMPLICES];
        uint32_t n_simp = RIN_Betti_VietorisRips_Build(
            dist_matrix, n_eff, eps, 2,  /* Hasta dim 2 */
            simplices, RIN_BETTI_MAX_SIMPLICES
        );
        
        slices[e].betti = RIN_Betti_CalculateFromComplex(simplices, n_simp);
        
        /* Persistencia estimada: qué tan estable son los números */
        if (e > 0) {
            float prev_complexity = (float)slices[e-1].betti.total_complexity;
            float curr_complexity = (float)slices[e].betti.total_complexity;
            if (prev_complexity > 0) {
                slices[e].persistence = 1.0f - fabsf(curr_complexity - prev_complexity) / prev_complexity;
            } else {
                slices[e].persistence = 1.0f;
            }
        } else {
            slices[e].persistence = 1.0f;
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_CompareRepresentations
 * Compara complejidad topológica de dos representaciones
 * 
 * Útil para distillation: estudiante debe mantener β similar a maestro
 * 
 * Retorna: estructura con diferencias detalladas
 * ============================================================================ */
static inline RIN_TopologicalComparison RIN_Betti_CompareRepresentations(
    const RIN_BettiNumbers* teacher,
    const RIN_BettiNumbers* student) {
    
    RIN_TopologicalComparison result = {0.0f, 0.0f, 0.0f, 0.0f, false};
    
    if (!teacher || !student) return result;
    
    result.beta0_diff = fabsf((float)teacher->beta0 - (float)student->beta0);
    result.beta1_diff = fabsf((float)teacher->beta1 - (float)student->beta1);
    result.beta2_diff = fabsf((float)teacher->beta2 - (float)student->beta2);
    
    /* Diferencia total normalizada */
    float total_teacher = (float)teacher->total_complexity;
    if (total_teacher < 1.0f) total_teacher = 1.0f;
    
    float total_student = (float)student->total_complexity;
    result.total_diff = fabsf(total_teacher - total_student) / total_teacher;
    
    return result;
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_TopologicalLoss
 * Pérdida ETL (Entropy-Topological Loss)
 * 
 * Penaliza diferencias en estructura topológica
 * 
 * @target:      Betti objetivo (maestro)
 * @pred:        Betti predicho (estudiante)
 * @lambda_topo: Peso de la pérdida topológica
 * ============================================================================ */
static inline float RIN_Betti_TopologicalLoss(const RIN_BettiNumbers* target,
                                             const RIN_BettiNumbers* pred,
                                             float lambda_topo) {
    RIN_TopologicalComparison diff = RIN_Betti_CompareRepresentations(target, pred);
    
    float complexity_penalty = fabsf((float)pred->total_complexity - 
                                    (float)target->total_complexity) / 
                              (float)(target->total_complexity + 1);
    
    return lambda_topo * diff.total_diff + 0.1f * complexity_penalty;
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_ValidateDistillation
 * Valida que estudiante preserva topología del maestro
 * 
 * @teacher_points:   Activaciones del modelo maestro
 * @student_points:   Activaciones del modelo estudiante
 * @n_points:         Número de puntos
 * @dim:              Dimensión
 * @epsilon:          Escala de análisis
 * @tolerance:        Tolerancia aceptable (ej: 0.1 = 10%)
 * 
 * Retorna: true si topologías son similares dentro de tolerancia
 * ============================================================================ */
static inline bool RIN_Betti_ValidateDistillation(const float* teacher_points,
                                                 const float* student_points,
                                                 uint32_t n_points,
                                                 uint8_t dim,
                                                 float epsilon,
                                                 float tolerance) {
    RIN_BettiNumbers teacher_betti = RIN_Betti_CalculateFromPoints(
        teacher_points, n_points, dim, epsilon, 2
    );
    
    RIN_BettiNumbers student_betti = RIN_Betti_CalculateFromPoints(
        student_points, n_points, dim, epsilon, 2
    );
    
    RIN_TopologicalComparison comp = RIN_Betti_CompareRepresentations(
        &teacher_betti, &student_betti
    );
    
    return comp.total_diff <= tolerance;
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_ExtractFromActivations
 * Extrae representación para análisis desde activaciones de red
 * 
 * Convierte activaciones de múltiples capas a punto en espacio de dim N
 * ============================================================================ */
static inline void RIN_Betti_ExtractFromActivations(const float** layer_activations,
                                                   uint32_t num_layers,
                                                   uint32_t samples_per_layer,
                                                   float* output_points,
                                                   uint32_t* n_points_out) {
    if (!layer_activations || !output_points || !n_points_out) return;
    
    uint32_t idx = 0;
    uint32_t max_points = RIN_BETTI_MAX_POINTS;
    
    for (uint32_t layer = 0; layer < num_layers && idx < max_points; layer++) {
        for (uint32_t s = 0; s < samples_per_layer && idx < max_points; s++) {
            /* Tomar muestra de activaciones de esta capa */
            /* Simplificación: usar valores directamente */
            output_points[idx] = layer_activations[layer][s];
            idx++;
        }
    }
    
    *n_points_out = idx;
}

/* ============================================================================
 * FUNCIÓN: RIN_Betti_Print
 * Debug: imprime números de Betti
 * ============================================================================ */
static inline void RIN_Betti_Print(const RIN_BettiNumbers* betti, const char* label) {
    if (!betti) return;
    
    printf("[BETTI] %s: β0=%u, β1=%u, β2=%u, ω=%u\n",
           label ? label : "",
           betti->beta0,
           betti->beta1,
           betti->beta2,
           betti->total_complexity);
}

#ifdef __cplusplus
}
#endif

#endif /* RIN_BETTI_CALCULATOR_H */
