/*
 * rin_dptm.h - Dynamic Parallel Thread Mapping (DPTM)
 * 
 * Mapeo Dinámico de Afinidad de CPU
 * Fija hilos a núcleos físicos específicos para evitar cache misses
 * 
 * Basado en: Linux sched.h + tud-zih-energy/x86_energy
 */

#ifndef RIN_DPTM_H
#define RIN_DPTM_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURACIÓN
 * ============================================================================ */

#ifndef RIN_DPTM_MAX_CORES
#define RIN_DPTM_MAX_CORES 256
#endif

/* ============================================================================
 * ENUMERACIONES
 * ============================================================================ */

/*
 * RIN_DPTM_AffinityPolicy - Estrategias de afinidad
 */
typedef enum {
    RIN_AFFINITY_SINGLE_CORE = 0,     /* Un hilo por núcleo, exclusivo */
    RIN_AFFINITY_SMT_AWARE = 1,       /* Respeta HyperThreading/SMT */
    RIN_AFFINITY_NUMA_LOCAL = 2,      /* NUMA-local allocation */
    RIN_AFFINITY_SCATTER = 3,         /* Distribución máxima */
    RIN_AFFINITY_DEFAULT = RIN_AFFINITY_SMT_AWARE
} RIN_DPTM_AffinityPolicy;

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_DPTM_CoreInfo - Información de topología detectada
 */
typedef struct {
    uint32_t physical_cores;    /* Núcleos físicos reales */
    uint32_t logical_cores;     /* Hilos lógicos (con SMT) */
    uint32_t numa_nodes;        /* Nodos NUMA */
    uint32_t l3_caches;         /* Caches L3 */
    uint32_t cores_per_l3;      /* Cores compartiendo L3 */
    uint32_t smt_factor;        /* 1 si no SMT, 2 si HyperThreading */
} RIN_DPTM_CoreInfo;

/*
 * RIN_DPTM_ThreadPool - Pool de hilos con afinidad fija
 */
typedef struct {
    pthread_t threads[RIN_DPTM_MAX_CORES];
    uint32_t num_threads;
    uint32_t core_mapping[RIN_DPTM_MAX_CORES];
    RIN_DPTM_AffinityPolicy policy;
} RIN_DPTM_ThreadPool;

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_Init
 * Detecta topología del sistema
 * 
 * @info: Estructura a poblar con información de topología
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
static inline int RIN_DPTM_Init(RIN_DPTM_CoreInfo* info) {
    if (!info) return -1;
    
    memset(info, 0, sizeof(RIN_DPTM_CoreInfo));
    
    /* Detección básica vía sysconf */
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs < 1) return -1;
    
    info->logical_cores = (uint32_t)nprocs;
    info->physical_cores = (uint32_t)nprocs; /* Asume sin SMT por default */
    info->numa_nodes = 1;
    info->l3_caches = 1;
    info->cores_per_l3 = (uint32_t)nprocs;
    info->smt_factor = 1;
    
    /* Intentar leer /proc/cpuinfo para físicos reales */
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        uint32_t cores = 0;
        uint32_t siblings = 0;
        
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "cpu cores", 9) == 0) {
                char* p = strchr(line, ':');
                if (p) cores = (uint32_t)atoi(p + 1);
            }
            if (strncmp(line, "siblings", 8) == 0) {
                char* p = strchr(line, ':');
                if (p) siblings = (uint32_t)atoi(p + 1);
            }
        }
        fclose(fp);
        
        if (cores > 0 && cores < info->logical_cores) {
            info->physical_cores = cores;
            info->smt_factor = siblings / cores;
        }
    }
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_SetAffinity
 * Fija hilo actual a núcleo específico
 * 
 * @core_id: ID del núcleo al que fijar el hilo
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
static inline int RIN_DPTM_SetAffinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_GetCurrentAffinity
 * Obtiene afinidad actual del hilo
 * 
 * @cpuset: CPU set a poblar
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
static inline int RIN_DPTM_GetCurrentAffinity(cpu_set_t* cpuset) {
    if (!cpuset) return -1;
    CPU_ZERO(cpuset);
    return pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuset);
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_GetCurrentCore
 * Obtiene núcleo actual donde ejecuta el hilo
 * 
 * Retorna: ID de núcleo (0-based), o -1 si fallo
 * ============================================================================ */
static inline int RIN_DPTM_GetCurrentCore(void) {
    return sched_getcpu();
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_YieldUltraLow
 * Yield de ultra-bajo consumo
 * 
 * Usa:
 *   - pause instruction en x86 (evita pipeline flush)
 *   - wfe (wait for event) en ARM
 *   - Fallback a sched_yield
 * ============================================================================ */
static inline void RIN_DPTM_YieldUltraLow(void) {
#if defined(__x86_64__) || defined(__i386__)
    /* pause instruction: ~10 ciclos, ahorra energía en spin-wait */
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    /* Wait for event - más eficiente que yield */
    __asm__ volatile("wfe" ::: "memory");
#elif defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    /* Fallback: sched_yield tiene overhead pero portable */
    sched_yield();
#endif
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_SpinWait
 * Espera activa eficiente (para buffers vacíos)
 * 
 * @iterations: número de spins antes de yield
 * ============================================================================ */
static inline void RIN_DPTM_SpinWait(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        RIN_DPTM_YieldUltraLow();
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_NanoSleep
 * Sleep de nanosegundos de alta precisión
 * 
 * @nanoseconds: tiempo a dormir
 * ============================================================================ */
static inline void RIN_DPTM_NanoSleep(uint64_t nanoseconds) {
    struct timespec ts = {
        .tv_sec = (time_t)(nanoseconds / 1000000000ULL),
        .tv_nsec = (long)(nanoseconds % 1000000000ULL)
    };
    nanosleep(&ts, NULL);
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_GetTimestampNs
 * Timestamp de alta resolución (nanosegundos)
 * 
 * Retorna: Timestamp monotónico en nanosegundos
 * ============================================================================ */
static inline uint64_t RIN_DPTM_GetTimestampNs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_GetTimestampUs
 * Timestamp en microsegundos
 * ============================================================================ */
static inline uint64_t RIN_DPTM_GetTimestampUs(void) {
    return RIN_DPTM_GetTimestampNs() / 1000ULL;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_GetTimestampMs
 * Timestamp en milisegundos
 * ============================================================================ */
static inline uint64_t RIN_DPTM_GetTimestampMs(void) {
    return RIN_DPTM_GetTimestampNs() / 1000000ULL;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_MeasureLatencyNs
 * Mide latencia de una operación
 * 
 * Uso:
 *   uint64_t start = RIN_DPTM_GetTimestampNs();
 *   // ... operación ...
 *   uint64_t latency = RIN_DPTM_MeasureLatencyNs(start);
 * ============================================================================ */
static inline uint64_t RIN_DPTM_MeasureLatencyNs(uint64_t start_ns) {
    return RIN_DPTM_GetTimestampNs() - start_ns;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_BusyWaitUntil
 * Espera ocupada hasta timestamp objetivo
 * 
 * @target_ns: timestamp objetivo
 * 
 * Retorna: 0 si llegó al target, -1 si ya pasó
 * ============================================================================ */
static inline int RIN_DPTM_BusyWaitUntil(uint64_t target_ns) {
    uint64_t now = RIN_DPTM_GetTimestampNs();
    if (now >= target_ns) return -1;
    
    /* Spin con yield hasta llegar */
    while (RIN_DPTM_GetTimestampNs() < target_ns) {
        RIN_DPTM_YieldUltraLow();
    }
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_ThreadPool_Init
 * Inicializa pool de hilos con afinidad fija
 * 
 * @pool:       Pool a inicializar
 * @num_threads: Número de hilos (<= physical_cores)
 * @policy:     Política de afinidad
 * @start_fn:   Función de inicio para cada hilo
 * @arg:        Argumento para start_fn
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
typedef void* (*RIN_DPTM_ThreadFunc)(void*);

static inline int RIN_DPTM_ThreadPool_Init(RIN_DPTM_ThreadPool* pool,
                                            uint32_t num_threads,
                                            RIN_DPTM_AffinityPolicy policy,
                                            RIN_DPTM_ThreadFunc start_fn,
                                            void* arg) {
    if (!pool || num_threads == 0 || num_threads > RIN_DPTM_MAX_CORES) return -1;
    
    RIN_DPTM_CoreInfo info;
    if (RIN_DPTM_Init(&info) != 0) return -1;
    
    /* Limitar a cores físicos si no queremos SMT */
    if (policy == RIN_AFFINITY_SINGLE_CORE && num_threads > info.physical_cores) {
        num_threads = info.physical_cores;
    }
    
    pool->num_threads = num_threads;
    pool->policy = policy;
    
    /* Crear hilos con afinidad */
    for (uint32_t i = 0; i < num_threads; i++) {
        /* Mapeo simple: hilo i -> core i */
        pool->core_mapping[i] = i;
        
        /* Crear hilo (simplificado - en producción pasaría core_id) */
        if (pthread_create(&pool->threads[i], NULL, start_fn, arg) != 0) {
            /* Cleanup hilos creados */
            for (uint32_t j = 0; j < i; j++) {
                pthread_cancel(pool->threads[j]);
            }
            return -1;
        }
        
        /* Fijar afinidad */
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        pthread_setaffinity_np(pool->threads[i], sizeof(cpu_set_t), &cpuset);
    }
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_DPTM_ThreadPool_Destroy
 * Destruye pool de hilos
 * ============================================================================ */
static inline void RIN_DPTM_ThreadPool_Destroy(RIN_DPTM_ThreadPool* pool) {
    if (!pool) return;
    
    for (uint32_t i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pool->num_threads = 0;
}

/* ============================================================================
 * MACROS DE CONVENIENCIA PARA PROFILING
 * ============================================================================ */

#define RIN_PROFILE_START() uint64_t _rin_prof_start = RIN_DPTM_GetTimestampNs()

#define RIN_PROFILE_END(label) do { \
    uint64_t _rin_prof_end = RIN_DPTM_GetTimestampNs(); \
    uint64_t _rin_prof_elapsed = _rin_prof_end - _rin_prof_start; \
    fprintf(stderr, "[PROFILE] %s: %lu ns (%.3f ms)\n", \
            label, _rin_prof_elapsed, (double)_rin_prof_elapsed / 1e6); \
} while(0)

#define RIN_PROFILE_SECTION(name, code) do { \
    RIN_PROFILE_START(); \
    code; \
    RIN_PROFILE_END(name); \
} while(0)

#ifdef __cplusplus
}
#endif

#endif /* RIN_DPTM_H */
