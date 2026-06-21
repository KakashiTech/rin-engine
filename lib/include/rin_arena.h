/*
 * rin_arena.h - Low Entropy Arena Allocator for RIN
 * 
 * O(1) allocation, O(1) reset, zero fragmentation
 * Based on: FurkanKirat/arena-allocator + ccgargantua/arena-allocator
 * 
 * ELIMINATES new/malloc during inference cycle
 * Only pointer bumping on pre-reserved contiguous block
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
 * CONFIGURATION
 * ============================================================================ */

#ifndef RIN_ARENA_DEFAULT_ALIGN
#define RIN_ARENA_DEFAULT_ALIGN 64  /* Cache line alignment */
#endif

#ifndef RIN_DEBUG
#define RIN_DEBUG 0
#endif

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/*
 * RIN_MemPool - Pre-reserved contiguous RAM block
 * Eliminates new/malloc during inference - only pointer bumping
 */
typedef struct {
    uint8_t* base;           /* Base pointer of the block */
    size_t   offset;         /* Current offset (bump pointer) */
    size_t   capacity;       /* Total capacity in bytes */
    size_t   high_watermark; /* Maximum usage for statistics */
} RIN_MemPool;

/*
 * RIN_MemoryArena - Collection of pools for different lifetime workloads
 */
typedef struct {
    RIN_MemPool inference;   /* Pool for inference cycle */
    RIN_MemPool scratch;     /* Per-frame temporary pool */
    RIN_MemPool persistent;  /* Pool for weights and bias */
} RIN_MemoryArena;

/* ============================================================================
 * FUNCTION: RIN_MemPool_Init
 * Initializes a pool with fixed capacity
 * 
 * @pool:      Pointer to pool structure
 * @capacity:  Bytes to pre-reserve
 * 
 * Returns: 0 on success, -1 on failure
 * ============================================================================ */
static inline int RIN_MemPool_Init(RIN_MemPool* pool, size_t capacity) {
    if (!pool || capacity == 0) return -1;
    
    /* Single malloc upfront - no kernel switches afterwards */
    pool->base = (uint8_t*)aligned_alloc(RIN_ARENA_DEFAULT_ALIGN, capacity);
    if (!pool->base) return -1;
    
    pool->offset = 0;
    pool->capacity = capacity;
    pool->high_watermark = 0;
    
    /* Prefetch hint for the allocator */
    __builtin_prefetch(pool->base, 1, 3);
    
    return 0;
}

/* ============================================================================
 * FUNCTION: RIN_MemPool_Alloc
 * Bump pointer allocation O(1)
 * 
 * @pool:   Pointer to pool
 * @size:   Bytes requested
 * @align:  Required alignment (power of 2)
 * 
 * Returns: Pointer to allocated block or NULL on failure
 * ============================================================================ */
static inline void* RIN_MemPool_Alloc(RIN_MemPool* pool, size_t size, size_t align) {
    if (!pool || size == 0) return NULL;
    
    /* Ensure align is a power of 2 */
    if (align & (align - 1)) align = RIN_ARENA_DEFAULT_ALIGN;
    
    /* Bitwise alignment (faster than modulo) */
    size_t mask = align - 1;
    size_t aligned_offset = (pool->offset + mask) & ~mask;
    size_t new_offset = aligned_offset + size;
    
    /* Bounds check - hard fail in debug, soft in release */
    if (new_offset > pool->capacity) {
#if RIN_DEBUG
        __builtin_trap(); /* Intentional crash for debugging */
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
 * FUNCTION: RIN_MemPool_Reset
 * Reset O(1) - frees everything at once
 * 
 * Does NOT release memory to OS - only resets offset for reuse
 * ============================================================================ */
static inline void RIN_MemPool_Reset(RIN_MemPool* pool) {
    if (pool) {
        pool->offset = 0;
        /* We don't touch high_watermark - useful for profiling */
    }
}

/* ============================================================================
 * FUNCTION: RIN_MemPool_Free
 * Complete pool deallocation (frees to OS)
 * ============================================================================ */
static inline void RIN_MemPool_Free(RIN_MemPool* pool) {
    if (pool && pool->base) {
        free(pool->base);
        pool->base = NULL;
        pool->offset = pool->capacity = pool->high_watermark = 0;
    }
}

/* ============================================================================
 * FUNCTION: RIN_MemPool_UsageRatio
 * Current usage ratio (0.0 - 1.0)
 * ============================================================================ */
static inline float RIN_MemPool_UsageRatio(const RIN_MemPool* pool) {
    return pool ? (float)pool->offset / (float)pool->capacity : 0.0f;
}

/* ============================================================================
 * FUNCTION: RIN_MemPool_SaveMarker
 * Saves current position for rollback
 * ============================================================================ */
static inline size_t RIN_MemPool_SaveMarker(const RIN_MemPool* pool) {
    return pool ? pool->offset : 0;
}

/* ============================================================================
 * FUNCTION: RIN_MemPool_Rollback
 * Rollback to saved position
 * ============================================================================ */
static inline void RIN_MemPool_Rollback(RIN_MemPool* pool, size_t marker) {
    if (pool && marker <= pool->offset) {
        pool->offset = marker;
    }
}

/* ============================================================================
 * FUNCTION: RIN_MemoryArena_Init
 * Initializes complete arena with 3 pools
 * 
 * @arena:              Pointer to arena
 * @inference_size:     Inference pool size
 * @scratch_size:       Scratch pool size
 * @persistent_size:    Persistent pool size
 * 
 * Returns: 0 on success, -1 on failure (partially initialized arena)
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
 * FUNCTION: RIN_MemoryArena_ResetInference
 * Reset only the inference pool (keeps weights)
 * ============================================================================ */
static inline void RIN_MemoryArena_ResetInference(RIN_MemoryArena* arena) {
    if (arena) {
        RIN_MemPool_Reset(&arena->inference);
        RIN_MemPool_Reset(&arena->scratch);
    }
}

/* ============================================================================
 * FUNCTION: RIN_MemoryArena_Destroy
 * Complete arena cleanup
 * ============================================================================ */
static inline void RIN_MemoryArena_Destroy(RIN_MemoryArena* arena) {
    if (arena) {
        RIN_MemPool_Free(&arena->inference);
        RIN_MemPool_Free(&arena->scratch);
        RIN_MemPool_Free(&arena->persistent);
    }
}

/* ============================================================================
 * CONVENIENCE MACROS
 * ============================================================================ */

/* Allocate a single object of type T in inference pool */
#define RIN_ALLOC(arena, T) \
    ((T*)RIN_MemPool_Alloc(&(arena)->inference, sizeof(T), alignof(T)))

/* Allocate array of N elements of type T in inference pool */
#define RIN_ALLOC_ARRAY(arena, T, N) \
    ((T*)RIN_MemPool_Alloc(&(arena)->inference, sizeof(T) * (N), alignof(T)))

/* Allocate array in scratch pool (temporary) */
#define RIN_SCRATCH_ALLOC(arena, T, N) \
    ((T*)RIN_MemPool_Alloc(&(arena)->scratch, sizeof(T) * (N), alignof(T)))

/* Allocate array in persistent pool (weights) */
#define RIN_PERSIST_ALLOC(arena, T, N) \
    ((T*)RIN_MemPool_Alloc(&(arena)->persistent, sizeof(T) * (N), alignof(T)))

/* Save marker and create RAII-style scope */
#define RIN_SCOPE_MARKER(arena, name) \
    size_t name = RIN_MemPool_SaveMarker(&(arena)->inference)

/* Rollback to marker */
#define RIN_SCOPE_ROLLBACK(arena, name) \
    RIN_MemPool_Rollback(&(arena)->inference, name)

#ifdef __cplusplus
}
#endif

#endif /* RIN_ARENA_H */
