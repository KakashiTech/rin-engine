/*
 * rin_energy_meter.h - Medición de Consumo Energético Real con Intel RAPL
 * 
 * Mide Joules exactos por inferencia usando:
 * - Intel RAPL (Running Average Power Limit) via MSRs
 * - Linux powercap sysfs (fallback)
 * - perf_event (alternativo)
 * 
 * Basado en: powercap/raplcap + sosy-lab/cpu-energy-meter
 * 
 * MÉTRICA CLAVE: Joules por cada 1000 tokens generados
 * Meta: < 0.5W por 1000 tokens = 20x vs PyTorch estándar (~10W)
 */

#ifndef RIN_ENERGY_METER_H
#define RIN_ENERGY_METER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "rin_dptm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTAS RAPL
 * ============================================================================ */

/* MSR addresses para RAPL */
#define MSR_RAPL_POWER_UNIT         0x606
#define MSR_PKG_ENERGY_STATUS       0x611
#define MSR_PKG_ENERGY_STATUS_MASK  0xFFFFFFFF

#define MSR_PP0_ENERGY_STATUS       0x639  /* Cores */
#define MSR_PP1_ENERGY_STATUS       0x641  /* GPU */
#define MSR_DRAM_ENERGY_STATUS      0x619  /* DRAM */
#define MSR_PLATFORM_ENERGY_STATUS  0x64D  /* Plataforma completa */

/* Dominios RAPL soportados */
typedef enum {
    RIN_RAPL_DOMAIN_PKG = 0,     /* Whole package - TODO el CPU */
    RIN_RAPL_DOMAIN_PP0,         /* Cores (Power Plane 0) */
    RIN_RAPL_DOMAIN_PP1,         /* GPU (Power Plane 1) */
    RIN_RAPL_DOMAIN_DRAM,        /* DRAM */
    RIN_RAPL_DOMAIN_PLATFORM,    /* Plataforma completa */
    RIN_RAPL_DOMAIN_MAX
} RIN_RAPL_Domain;

/* Nombres para debug */
static const char* RIN_RAPL_DomainNames[] = {
    "PKG", "PP0", "PP1", "DRAM", "PLATFORM"
};

/* ============================================================================
 * ESTRUCTURAS DE DATOS
 * ============================================================================ */

/*
 * RIN_EnergyReading - Lectura de energía
 */
typedef struct {
    uint64_t joules_raw;         /* Valor raw del MSR (32 bits, wraparound) */
    double   joules;             /* Convertido a Joules */
    double   watts;              /* Potencia instantánea si disponible */
    uint64_t timestamp_ns;       /* Timestamp de lectura (monotónico) */
    uint32_t overflow_count;     /* Contador de wraparounds detectados */
} RIN_EnergyReading;

/*
 * RIN_EnergyMeter - Estado del medidor
 */
typedef struct {
    int msr_fds[RIN_RAPL_DOMAIN_MAX];      /* File descriptors para MSRs */
    double energy_units[RIN_RAPL_DOMAIN_MAX];  /* Multiplicador para convertir a Joules */
    double power_units[RIN_RAPL_DOMAIN_MAX];   /* Multiplicador para convertir a Watts */
    double time_units[RIN_RAPL_DOMAIN_MAX];    /* Multiplicador para segundos */
    
    uint64_t max_energy_status[RIN_RAPL_DOMAIN_MAX];  /* Para detectar wraparound */
    uint64_t last_raw_reading[RIN_RAPL_DOMAIN_MAX];   /* Lectura anterior */
    
    uint32_t num_packages;       /* Número de packages detectados */
    uint32_t initialized;        /* Flag de inicialización */
    
    /* Modo de operación */
    uint8_t use_sysfs;           /* 1=usa sysfs, 0=usa MSR directo */
    uint8_t has_rapl;            /* 1=RAPL disponible, 0=no disponible */
} RIN_EnergyMeter;

/*
 * RIN_EnergyMetrics - Métricas derivadas
 */
typedef struct {
    double joules_per_token;     /* Energía por token */
    double joules_per_1k_tokens; /* Energía por 1000 tokens */
    double watts_average;        /* Potencia promedio */
    double watts_per_1k;         /* Potencia equivalente por 1000 tokens */
    double tokens_per_joule;     /* Tokens procesados por Joule */
    uint64_t inference_time_ns;  /* Tiempo total de inferencia */
    uint32_t tokens_processed;   /* Tokens procesados */
} RIN_EnergyMetrics;

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_CheckRAPLSupport
 * Verifica si RAPL está disponible en el sistema
 * ============================================================================ */
static inline int RIN_EnergyMeter_CheckRAPLSupport(void) {
    /* Intentar abrir MSR del package 0 */
    char msr_path[64];
    snprintf(msr_path, sizeof(msr_path), "/dev/cpu/0/msr");
    
    int fd = open(msr_path, O_RDONLY);
    if (fd >= 0) {
        close(fd);
        return 1;  /* MSR disponible */
    }
    
    /* Fallback: verificar sysfs powercap */
    FILE* fp = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
    if (fp) {
        fclose(fp);
        return 2;  /* sysfs disponible */
    }
    
    return 0;  /* RAPL no disponible */
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_ReadMSR
 * Lee MSR directamente
 * ============================================================================ */
static inline uint64_t RIN_EnergyMeter_ReadMSR(int fd, uint32_t msr) {
    uint64_t value = 0;
    
    if (pread(fd, &value, sizeof(uint64_t), msr) != sizeof(uint64_t)) {
        return 0;
    }
    
    return value;
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_Init
 * Inicializa acceso a RAPL
 * 
 * Intenta primero MSR directo, luego fallback a sysfs
 * 
 * Retorna: 0 si éxito, -1 si RAPL no disponible
 * ============================================================================ */
static inline int RIN_EnergyMeter_Init(RIN_EnergyMeter* meter) {
    if (!meter) return -1;
    
    memset(meter, 0, sizeof(RIN_EnergyMeter));
    
    /* Verificar soporte */
    int rapl_support = RIN_EnergyMeter_CheckRAPLSupport();
    if (rapl_support == 0) {
        return -1;  /* RAPL no disponible */
    }
    
    meter->has_rapl = 1;
    
    if (rapl_support == 2) {
        /* Usar modo sysfs */
        meter->use_sysfs = 1;
        meter->initialized = 1;
        meter->num_packages = 1;
        
        /* Unidades típicas para sysfs (microJoules) */
        for (int i = 0; i < RIN_RAPL_DOMAIN_MAX; i++) {
            meter->energy_units[i] = 1.0e-6;  /* uJ -> J */
        }
        
        return 0;
    }
    
    /* Modo MSR directo */
    char msr_path[64];
    snprintf(msr_path, sizeof(msr_path), "/dev/cpu/0/msr");
    
    int fd = open(msr_path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    
    /* Leer unidades de energía del MSR_RAPL_POWER_UNIT */
    uint64_t power_unit_msr = RIN_EnergyMeter_ReadMSR(fd, MSR_RAPL_POWER_UNIT);
    close(fd);
    
    if (power_unit_msr == 0) {
        return -1;
    }
    
    /* Extraer unidades según Intel SDM:
     * bits 8:14 = energy unit (por defecto 15.3 micro-Joules si 0x1E = 30)
     * bits 0:3  = power unit  
     * bits 16:22 = time unit
     */
    uint8_t energy_unit_exponent = (power_unit_msr >> 8) & 0x1F;
    uint8_t power_unit_exponent = (power_unit_msr >> 0) & 0x0F;
    uint8_t time_unit_exponent = (power_unit_msr >> 16) & 0x1F;
    
    /* Energy unit: 1 / (2^exponent) Joules */
    double energy_unit = 1.0 / (1 << energy_unit_exponent);
    if (energy_unit_exponent == 0) energy_unit = 1.0;  /* Fallback */
    
    /* Power unit: 1 / (2^exponent) Watts */
    double power_unit = 1.0 / (1 << power_unit_exponent);
    if (power_unit_exponent == 0) power_unit = 1.0;
    
    /* Time unit: 1 / (2^exponent) Seconds */
    double time_unit = 1.0 / (1 << time_unit_exponent);
    if (time_unit_exponent == 0) time_unit = 1.0;
    
    for (int i = 0; i < RIN_RAPL_DOMAIN_MAX; i++) {
        meter->energy_units[i] = energy_unit;
        meter->power_units[i] = power_unit;
        meter->time_units[i] = time_unit;
        
        /* Max energy status antes de wraparound (32 bits) */
        meter->max_energy_status[i] = (uint64_t)(energy_unit * ((1ULL << 32) - 1));
    }
    
    /* Abrir FDs para cada dominio (solo PKG por ahora) */
    meter->msr_fds[RIN_RAPL_DOMAIN_PKG] = open(msr_path, O_RDONLY);
    if (meter->msr_fds[RIN_RAPL_DOMAIN_PKG] < 0) {
        return -1;
    }
    
    meter->initialized = 1;
    meter->num_packages = 1;
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_ReadDomain
 * Lee energía consumida de un dominio específico
 * 
 * @domain: Dominio a leer (PKG, PP0, DRAM, etc.)
 * @reading: Estructura a poblar
 * 
 * Retorna: 0 si éxito, -1 si fallo
 * ============================================================================ */
static inline int RIN_EnergyMeter_ReadDomain(RIN_EnergyMeter* meter,
                                              RIN_RAPL_Domain domain,
                                              RIN_EnergyReading* reading) {
    if (!meter || !meter->initialized || !reading) return -1;
    if (domain >= RIN_RAPL_DOMAIN_MAX) return -1;
    
    reading->timestamp_ns = RIN_DPTM_GetTimestampNs();
    
    if (meter->use_sysfs) {
        /* Modo sysfs */
        char path[128];
        const char* domain_name = "intel-rapl:0";
        const char* subname = "";
        
        switch (domain) {
            case RIN_RAPL_DOMAIN_PKG:  subname = ""; break;
            case RIN_RAPL_DOMAIN_PP0:  subname = "/intel-rapl:0:0"; break;
            case RIN_RAPL_DOMAIN_DRAM: subname = "/intel-rapl:0:1"; break;
            default: subname = ""; break;
        }
        
        snprintf(path, sizeof(path), 
                "/sys/class/powercap/%s%s/energy_uj",
                domain_name, subname);
        
        FILE* fp = fopen(path, "r");
        if (!fp) return -1;
        
        uint64_t uj;
        if (fscanf(fp, "%lu", &uj) != 1) {
            fclose(fp);
            return -1;
        }
        fclose(fp);
        
        reading->joules_raw = uj;
        reading->joules = (double)uj / 1000000.0;  /* uJ -> J */
        reading->watts = 0.0;  /* No disponible directamente en sysfs */
        
    } else {
        /* Modo MSR directo */
        if (meter->msr_fds[RIN_RAPL_DOMAIN_PKG] < 0) return -1;
        
        uint32_t msr_addr;
        switch (domain) {
            case RIN_RAPL_DOMAIN_PKG:  msr_addr = MSR_PKG_ENERGY_STATUS; break;
            case RIN_RAPL_DOMAIN_PP0:  msr_addr = MSR_PP0_ENERGY_STATUS; break;
            case RIN_RAPL_DOMAIN_PP1:  msr_addr = MSR_PP1_ENERGY_STATUS; break;
            case RIN_RAPL_DOMAIN_DRAM: msr_addr = MSR_DRAM_ENERGY_STATUS; break;
            default: return -1;
        }
        
        uint64_t energy_msr = RIN_EnergyMeter_ReadMSR(
            meter->msr_fds[RIN_RAPL_DOMAIN_PKG], msr_addr
        );
        
        /* Mask 32 bits (RAPL usa solo los primeros 32 bits) */
        reading->joules_raw = energy_msr & MSR_PKG_ENERGY_STATUS_MASK;
        reading->joules = (double)reading->joules_raw * meter->energy_units[domain];
        
        /* Detectar wraparound */
        if (reading->joules_raw < meter->last_raw_reading[domain]) {
            reading->overflow_count = 1;  /* Detectado wraparound */
            /* Ajustar: asumir que pasamos por el máximo */
            reading->joules += meter->max_energy_status[domain];
        } else {
            reading->overflow_count = 0;
        }
        
        meter->last_raw_reading[domain] = reading->joules_raw;
    }
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_StartMeasurement
 * Inicia medición de energía (guarda baseline)
 * ============================================================================ */
typedef struct {
    RIN_EnergyReading start_readings[RIN_RAPL_DOMAIN_MAX];
    uint64_t start_time_ns;
} RIN_EnergyMeasurement;

static inline int RIN_EnergyMeter_StartMeasurement(RIN_EnergyMeter* meter,
                                                    RIN_EnergyMeasurement* meas) {
    if (!meter || !meas) return -1;
    
    meas->start_time_ns = RIN_DPTM_GetTimestampNs();
    
    /* Leer todos los dominios disponibles */
    for (int d = 0; d < RIN_RAPL_DOMAIN_MAX; d++) {
        RIN_EnergyMeter_ReadDomain(meter, d, &meas->start_readings[d]);
    }
    
    return 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_EndMeasurement
 * Finaliza medición y calcula delta
 * 
 * Retorna: energía consumida en Joules (dominio PKG por defecto)
 * ============================================================================ */
static inline double RIN_EnergyMeter_EndMeasurement(RIN_EnergyMeter* meter,
                                                   RIN_EnergyMeasurement* meas,
                                                   RIN_RAPL_Domain domain) {
    if (!meter || !meas) return -1.0;
    
    RIN_EnergyReading end_reading;
    if (RIN_EnergyMeter_ReadDomain(meter, domain, &end_reading) != 0) {
        return -1.0;
    }
    
    double joules_consumed = end_reading.joules - meas->start_readings[domain].joules;
    if (joules_consumed < 0) joules_consumed = 0;  /* Sanity check */
    
    return joules_consumed;
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_ComputeMetrics
 * Computa métricas derivadas
 * 
 * @joules_consumed: Energía total medida
 * @tokens_processed: Tokens generados/procesados
 * @time_ns: Tiempo transcurrido en nanosegundos
 * @metrics: Estructura a poblar
 * ============================================================================ */
static inline void RIN_EnergyMeter_ComputeMetrics(double joules_consumed,
                                                  uint32_t tokens_processed,
                                                  uint64_t time_ns,
                                                  RIN_EnergyMetrics* metrics) {
    if (!metrics) return;
    
    memset(metrics, 0, sizeof(RIN_EnergyMetrics));
    
    metrics->tokens_processed = tokens_processed;
    metrics->inference_time_ns = time_ns;
    
    if (tokens_processed > 0) {
        metrics->joules_per_token = joules_consumed / (double)tokens_processed;
        metrics->joules_per_1k_tokens = metrics->joules_per_token * 1000.0;
        metrics->tokens_per_joule = (double)tokens_processed / joules_consumed;
    }
    
    if (time_ns > 0) {
        double time_seconds = (double)time_ns / 1e9;
        metrics->watts_average = joules_consumed / time_seconds;
        
        /* Potencia equivalente por 1000 tokens */
        double tokens_per_sec = (double)tokens_processed / time_seconds;
        if (tokens_per_sec > 0) {
            double seconds_per_1k = 1000.0 / tokens_per_sec;
            metrics->watts_per_1k = metrics->watts_average;  /* Misma potencia, escalada */
        }
    }
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_Close
 * Limpieza
 * ============================================================================ */
static inline void RIN_EnergyMeter_Close(RIN_EnergyMeter* meter) {
    if (!meter) return;
    
    for (int i = 0; i < RIN_RAPL_DOMAIN_MAX; i++) {
        if (meter->msr_fds[i] > 0) {
            close(meter->msr_fds[i]);
            meter->msr_fds[i] = -1;
        }
    }
    
    meter->initialized = 0;
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_PrintMetrics
 * Imprime métricas formateadas
 * ============================================================================ */
static inline void RIN_EnergyMeter_PrintMetrics(const RIN_EnergyMetrics* metrics) {
    if (!metrics) return;
    
    printf("\n========== RIN ENERGY METRICS ==========\n");
    printf("Tokens processed:     %u\n", metrics->tokens_processed);
    printf("Time elapsed:         %.3f ms\n", (double)metrics->inference_time_ns / 1e6);
    printf("Joules per token:     %.6f J\n", metrics->joules_per_token);
    printf("Joules per 1K tokens: %.3f J\n", metrics->joules_per_1k_tokens);
    printf("Watts average:        %.3f W\n", metrics->watts_average);
    printf("Tokens per Joule:     %.2f\n", metrics->tokens_per_joule);
    printf("========================================\n\n");
}

/* ============================================================================
 * FUNCIÓN: RIN_EnergyMeter_ValidateTarget
 * Valida si se cumple meta de eficiencia
 * 
 * Meta: < 0.5W por 1000 tokens
 * ============================================================================ */
static inline bool RIN_EnergyMeter_ValidateTarget(const RIN_EnergyMetrics* metrics,
                                                 float target_watts_per_1k) {
    if (!metrics) return false;
    
    return metrics->joules_per_1k_tokens <= target_watts_per_1k;
}

/* ============================================================================
 * MACRO: RIN_ENERGY_MEASURE_SCOPE
 * Macro para medir automáticamente un bloque de código
 * 
 * Uso:
 *   RIN_ENERGY_MEASURE_SCOPE(meter, metrics, {
 *       // ... código a medir ...
 *   });
 * ============================================================================ */
#define RIN_ENERGY_MEASURE_SCOPE(meter, metrics_var, code) \
    do { \
        RIN_EnergyMeasurement _meas; \
        RIN_EnergyMeter_StartMeasurement((meter), &_meas); \
        code; \
        uint64_t _end_time = RIN_DPTM_GetTimestampNs(); \
        double _joules = RIN_EnergyMeter_EndMeasurement((meter), &_meas, RIN_RAPL_DOMAIN_PKG); \
        uint64_t _time_ns = _end_time - _meas.start_time_ns; \
        /* Asume que tokens se cuentan externamente */ \
        RIN_EnergyMeter_ComputeMetrics(_joules, 0, _time_ns, &(metrics_var)); \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* RIN_ENERGY_METER_H */
