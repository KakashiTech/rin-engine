/*
 * rin_arena.h - Arena Allocator de Baja Entropía para RIN
 * 
 * O(1) allocation, O(1) reset, zero fragmentation
 * Basado en: FurkanKirat/arena-allocator + ccgargantua/arena-allocator
 * 
 * ELIMINA new/malloc durante ciclo de inferencia
 * Solo pointer bumping sobre bloque contiguo pre-reservado
 */

#ifndef RIN_ARENA_H
#define RIN_ARENA_H

#include <stdint.h>
#include <stddef.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURACIÓN
 * ============================================================================ */

#ifndef RIN_ARENA_DEFAULT_ALIGN
#define RIN_ARENA_DEFAULT_ALIGN 64  /* Cache line alignment */
#endif

#ifndef RIN_DEBUG
#define RIN_DEBUG 0
#endif

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_MemPool - Bloque contiguo de RAM pre-reservado
 * Eliminamos new/malloc durante inferencia - solo pointer bumping
 */
typedef struct {
    uint8_t* base;           /* Puntero base del bloque */
    size_t   offset;         /* Offset actual (bump pointer) */
    size_t   capacity;       /* Capacidad total en bytes */
    size_t   high_watermark; /* Máximo uso para estadísticas */
} RIN_MemPool;

/*
 * RIN_MemoryArena - Colección de pools para workloads de diferentes lifetimes
 */
typedef struct {
    RIN_MemPool inference;   /* Pool para ciclo de inferencia */
    RIN_MemPool scratch;     /* Pool temporal por-frame */
    RIN_MemPool persistent;  /* Pool para pesos y bias */
} RIN_MemoryArena;

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_Init
 * Inicializa un pool con capacidad fija
 * 
 * @pool:      Puntero a estructura pool
 * @capacity:  Bytes a pre-reservar
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
static inline int RIN_MemPool_Init(RIN_MemPool* pool, size_t capacity) {
    if (!pool || capacity == 0) return -1;
    
    /* Single malloc upfront - no kernel switches después */
    pool->base = (uint8_t*)aligned_alloc(RIN_ARENA_DEFAULT_ALIGN, capacity);
    if (!pool->base) return -1;
    
    pool->offset = 0;
    pool->capacity = capacity;
    pool->high_watermark = 0;
    
    /* Prefetch hint para el allocator */
    __builtin_prefetch(pool->base, 1, 3);
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_Alloc
 * Bump pointer allocation O(1)
 * 
 * @pool:   Puntero al pool
 * @size:   Bytes solicitados
 * @align:  Alineación requerida (potencia de 2)
 * 
 * Retorna: Puntero al bloque asignado o NULL si falla
 * ============================================================================ */
static inline void* RIN_MemPool_Alloc(RIN_MemPool* pool, size_t size, size_t align) {
    if (!pool || size == 0) return NULL;
    
    /* Asegurar que align es potencia de 2 */
    if (align & (align - 1)) align = RIN_ARENA_DEFAULT_ALIGN;
    
    /* Alineación bitwise (más rápido que modulo) */
    size_t mask = align - 1;
    size_t aligned_offset = (pool->offset + mask) & ~mask;
    size_t new_offset = aligned_offset + size;
    
    /* Bounds check - hard fail en debug, soft en release */
    if (new_offset > pool->capacity) {
#if RIN_DEBUG
        __builtin_trap(); /* Intencional crash para debugging */
#else
        return NULL;
#endif
    }
    
    void* ptr = pool->base + aligned_offset;
    pool->offset = new_offset;
    
    if (new_offset > pool->high_watermark) {
        pool->high_watermark = new_offset;
    }
    
    return ptr;
}

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_Reset
 * Reset O(1) - libera TODO de una vez
 * 
 * NO libera memoria al OS - solo resetea offset para reutilización
 * ============================================================================ */
static inline void RIN_MemPool_Reset(RIN_MemPool* pool) {
    if (pool) {
        pool->offset = 0;
        /* No tocamos high_watermark - útil para profiling */
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_Free
 * Liberación completa del pool (libera al OS)
 * ============================================================================ */
static inline void RIN_MemPool_Free(RIN_MemPool* pool) {
    if (pool && pool->base) {
        free(pool->base);
        pool->base = NULL;
        pool->offset = pool->capacity = pool->high_watermark = 0;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_UsageRatio
 * Ratio de uso actual (0.0 - 1.0)
 * ============================================================================ */
static inline float RIN_MemPool_UsageRatio(const RIN_MemPool* pool) {
    return pool ? (float)pool->offset / (float)pool->capacity : 0.0f;
}

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_SaveMarker
 * Guarda posición actual para rollback
 * ============================================================================ */
static inline size_t RIN_MemPool_SaveMarker(const RIN_MemPool* pool) {
    return pool ? pool->offset : 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_MemPool_Rollback
 * Rollback a posición guardada
 * ============================================================================ */
static inline void RIN_MemPool_Rollback(RIN_MemPool* pool, size_t marker) {
    if (pool && marker <= pool->offset) {
        pool->offset = marker;
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_MemoryArena_Init
 * Inicializa arena completa con 3 pools
 * 
 * @arena:              Puntero a arena
 * @inference_size:     Tamaño pool de inferencia
 * @scratch_size:       Tamaño pool scratch
 * @persistent_size:    Tamaño pool persistente
 * 
 * Retorna: 0 si éxito, -1 si fallo (arena parcialmente inicializada)
 * ============================================================================ */
static inline int RIN_MemoryArena_Init(RIN_MemoryArena* arena,
                                        size_t inference_size,
                                        size_t scratch_size,
                                        size_t persistent_size) {
    if (!arena) return -1;
    
    memset(arena, 0, sizeof(RIN_MemoryArena));
    
    if (RIN_MemPool_Init(&arena->inference, inference_size) != 0) return -1;
    if (RIN_MemPool_Init(&arena->scratch, scratch_size) != 0) {
        RIN_MemPool_Free(&arena->inference);
        return -1;
    }
    if (RIN_MemPool_Init(&arena->persistent, persistent_size) != 0) {
        RIN_MemPool_Free(&arena->inference);
        RIN_MemPool_Free(&arena->scratch);
        return -1;
    }
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_MemoryArena_ResetInference
 * Reset solo del pool de inferencia (conserva pesos)
 * ============================================================================ */
static inline void RIN_MemoryArena_ResetInference(RIN_MemoryArena* arena) {
    if (arena) {
        RIN_MemPool_Reset(&arena->inference);
        RIN_MemPool_Reset(&arena->scratch);
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_MemoryArena_Destroy
 * Limpieza completa de arena
 * ============================================================================ */
static inline void RIN_MemoryArena_Destroy(RIN_MemoryArena* arena) {
    if (arena) {
        RIN_MemPool_Free(&arena->inference);
        RIN_MemPool_Free(&arena->scratch);
        RIN_MemPool_Free(&arena->persistent);
    }
}

/* ============================================================================
 * MACROS DE CONVENIENCIA
 * ============================================================================ */

/* Allocar un solo objeto de tipo T en pool de inferencia */
#define RIN_ALLOC(arena, T) \
    ((T*)RIN_MemPool_Alloc(&(arena)->inference, sizeof(T), alignof(T)))

/* Allocar array de N elementos de tipo T en pool de inferencia */
#define RIN_ALLOC_ARRAY(arena, T, N) \
    ((T*)RIN_MemPool_Alloc(&(arena)->inference, sizeof(T) * (N), alignof(T)))

/* Allocar array en pool scratch (temporal) */
#define RIN_SCRATCH_ALLOC(arena, T, N) \
    ((T*)RIN_MemPool_Alloc(&(arena)->scratch, sizeof(T) * (N), alignof(T)))

/* Allocar array en pool persistente (pesos) */
#define RIN_PERSIST_ALLOC(arena, T, N) \
    ((T*)RIN_MemPool_Alloc(&(arena)->persistent, sizeof(T) * (N), alignof(T)))

/* Guardar marker y crear scope RAII-style */
#define RIN_SCOPE_MARKER(arena, name) \
    size_t name = RIN_MemPool_SaveMarker(&(arena)->inference)

/* Rollback al marker */
#define RIN_SCOPE_ROLLBACK(arena, name) \
    RIN_MemPool_Rollback(&(arena)->inference, name)

#ifdef __cplusplus
}
#endif

#endif /* RIN_ARENA_H */
