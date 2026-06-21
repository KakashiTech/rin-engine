/*
 * rin_energy_meter.h - Real Energy Consumption Measurement with Intel RAPL
 * 
 * Measures exact Joules per inference using:
 * - Intel RAPL (Running Average Power Limit) via MSRs
 * - Linux powercap sysfs (fallback)
 * - perf_event (alternative)
 * 
 * Based on: powercap/raplcap + sosy-lab/cpu-energy-meter
 * 
 * KEY METRIC: Joules per 1000 generated tokens
 * Target: < 0.5W per 1000 tokens = 20x vs standard PyTorch (~10W)
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
 * RAPL CONSTANTS
 * ============================================================================ */

/* MSR addresses for RAPL */
#define MSR_RAPL_POWER_UNIT         0x606
#define MSR_PKG_ENERGY_STATUS       0x611
#define MSR_PKG_ENERGY_STATUS_MASK  0xFFFFFFFF

#define MSR_PP0_ENERGY_STATUS       0x639  /* Cores */
#define MSR_PP1_ENERGY_STATUS       0x641  /* GPU */
#define MSR_DRAM_ENERGY_STATUS      0x619  /* DRAM */
#define MSR_PLATFORM_ENERGY_STATUS  0x64D  /* Complete platform */

/* Supported RAPL domains */
typedef enum {
    RIN_RAPL_DOMAIN_PKG = 0,     /* Whole package - TODO el CPU */
    RIN_RAPL_DOMAIN_PP0,         /* Cores (Power Plane 0) */
    RIN_RAPL_DOMAIN_PP1,         /* GPU (Power Plane 1) */
    RIN_RAPL_DOMAIN_DRAM,        /* DRAM */
    RIN_RAPL_DOMAIN_PLATFORM,    /* Complete platform */
    RIN_RAPL_DOMAIN_MAX
} RIN_RAPL_Domain;

/* Names for debug */
static const char* RIN_RAPL_DomainNames[] = {
    "PKG", "PP0", "PP1", "DRAM", "PLATFORM"
};

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/*
 * RIN_EnergyReading - Energy reading
 */
typedef struct {
    uint64_t joules_raw;         /* Raw MSR value (32 bits, wraparound) */
    double   joules;             /* Converted to Joules */
    double   watts;              /* Instantaneous power if available */
    uint64_t timestamp_ns;       /* Read timestamp (monotonic) */
    uint32_t overflow_count;     /* Count of detected wraparounds */
} RIN_EnergyReading;

/*
 * RIN_EnergyMeter - Meter state
 */
typedef struct {
    int msr_fds[RIN_RAPL_DOMAIN_MAX];      /* File descriptors for MSRs */
    double energy_units[RIN_RAPL_DOMAIN_MAX];  /* Multiplier to convert to Joules */
    double power_units[RIN_RAPL_DOMAIN_MAX];   /* Multiplier to convert to Watts */
    double time_units[RIN_RAPL_DOMAIN_MAX];    /* Multiplier for seconds */
    
    uint64_t max_energy_status[RIN_RAPL_DOMAIN_MAX];  /* For wraparound detection */
    uint64_t last_raw_reading[RIN_RAPL_DOMAIN_MAX];   /* Previous reading */
    
    uint32_t num_packages;       /* Number of detected packages */
    uint32_t initialized;        /* Initialization flag */
    
    /* Operation mode */
    uint8_t use_sysfs;           /* 1=use sysfs, 0=use direct MSR */
    uint8_t has_rapl;            /* 1=RAPL available, 0=not available */
} RIN_EnergyMeter;

/*
 * RIN_EnergyMetrics - Derived metrics
 */
typedef struct {
    double joules_per_token;     /* Energy per token */
    double joules_per_1k_tokens; /* Energy per 1000 tokens */
    double watts_average;        /* Average power */
    double watts_per_1k;         /* Equivalent power per 1000 tokens */
    double tokens_per_joule;     /* Tokens processed per Joule */
    uint64_t inference_time_ns;  /* Total inference time */
    uint32_t tokens_processed;   /* Tokens processed */
} RIN_EnergyMetrics;

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_CheckRAPLSupport
 * Checks if RAPL is available on the system
 * ============================================================================ */
static inline int RIN_EnergyMeter_CheckRAPLSupport(void) {
    /* Try to open MSR of package 0 */
    char msr_path[64];
    snprintf(msr_path, sizeof(msr_path), "/dev/cpu/0/msr");
    
    int fd = open(msr_path, O_RDONLY);
    if (fd >= 0) {
        close(fd);
        return 1;  /* MSR available */
    }
    
    /* Fallback: check sysfs powercap */
    FILE* fp = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
    if (fp) {
        fclose(fp);
        return 2;  /* sysfs available */
    }
    
    return 0;  /* RAPL not available */
}

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_ReadMSR
 * Reads MSR directly
 * ============================================================================ */
static inline uint64_t RIN_EnergyMeter_ReadMSR(int fd, uint32_t msr) {
    uint64_t value = 0;
    
    if (pread(fd, &value, sizeof(uint64_t), msr) != sizeof(uint64_t)) {
        return 0;
    }
    
    return value;
}

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_Init
 * Initializes RAPL access
 * 
 * Tries direct MSR first, then falls back to sysfs
 * 
 * Returns: 0 on success, -1 if RAPL not available
 * ============================================================================ */
static inline int RIN_EnergyMeter_Init(RIN_EnergyMeter* meter) {
    if (!meter) return -1;
    
    memset(meter, 0, sizeof(RIN_EnergyMeter));
    
    /* Check support */
    int rapl_support = RIN_EnergyMeter_CheckRAPLSupport();
    if (rapl_support == 0) {
        return -1;  /* RAPL not available */
    }
    
    meter->has_rapl = 1;
    
    if (rapl_support == 2) {
        /* Use sysfs mode */
        meter->use_sysfs = 1;
        meter->initialized = 1;
        meter->num_packages = 1;
        
        /* Typical units for sysfs (microJoules) */
        for (int i = 0; i < RIN_RAPL_DOMAIN_MAX; i++) {
            meter->energy_units[i] = 1.0e-6;  /* uJ -> J */
        }
        
        return 0;
    }
    
        /* Direct MSR mode */
    char msr_path[64];
    snprintf(msr_path, sizeof(msr_path), "/dev/cpu/0/msr");
    
    int fd = open(msr_path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    
    /* Read energy units from MSR_RAPL_POWER_UNIT */
    uint64_t power_unit_msr = RIN_EnergyMeter_ReadMSR(fd, MSR_RAPL_POWER_UNIT);
    close(fd);
    
    if (power_unit_msr == 0) {
        return -1;
    }
    
    /* Extract units per Intel SDM:
     * bits 8:14 = energy unit (default 15.3 micro-Joules if 0x1E = 30)
     * bits 0:3  = power unit  
     * bits 16:22 = time unit
     */
    uint8_t energy_unit_exponent = (power_unit_msr >> 8) & 0x1F;
    uint8_t power_unit_exponent = (power_unit_msr >> 0) & 0x0F;
    uint8_t time_unit_exponent = (power_unit_msr >> 16) & 0x1F;
    
    /* Energy unit: 1 / (2^exponent) Joules */
    double energy_unit = 1.0 / (1 << energy_unit_exponent);
    if (energy_unit_exponent == 0) energy_unit = 1.0;  /* fallback */
    
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
        
        /* Max energy status before wraparound (32 bits) */
        meter->max_energy_status[i] = (uint64_t)(energy_unit * ((1ULL << 32) - 1));
    }
    
    /* Open FDs for each domain (only PKG for now) */
    meter->msr_fds[RIN_RAPL_DOMAIN_PKG] = open(msr_path, O_RDONLY);
    if (meter->msr_fds[RIN_RAPL_DOMAIN_PKG] < 0) {
        return -1;
    }
    
    meter->initialized = 1;
    meter->num_packages = 1;
    
    return 0;
}

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_ReadDomain
 * Reads energy consumed by a specific domain
 * 
 * @domain: Domain to read (PKG, PP0, DRAM, etc.)
 * @reading: Structure to populate
 * 
 * Returns: 0 on success, -1 on failure
 * ============================================================================ */
static inline int RIN_EnergyMeter_ReadDomain(RIN_EnergyMeter* meter,
                                              RIN_RAPL_Domain domain,
                                              RIN_EnergyReading* reading) {
    if (!meter || !meter->initialized || !reading) return -1;
    if (domain >= RIN_RAPL_DOMAIN_MAX) return -1;
    
    reading->timestamp_ns = RIN_DPTM_GetTimestampNs();
    
    if (meter->use_sysfs) {
        /* Sysfs mode */
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
        reading->watts = 0.0;  /* Not directly available in sysfs */
        
    } else {
    /* Direct MSR mode */
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
        
        /* Mask 32 bits (RAPL uses only the first 32 bits) */
        reading->joules_raw = energy_msr & MSR_PKG_ENERGY_STATUS_MASK;
        reading->joules = (double)reading->joules_raw * meter->energy_units[domain];
        
        /* Detect wraparound */
        if (reading->joules_raw < meter->last_raw_reading[domain]) {
            reading->overflow_count = 1;  /* Detected wraparound */
            /* Adjust: assume we passed through the maximum */
            reading->joules += meter->max_energy_status[domain];
        } else {
            reading->overflow_count = 0;
        }
        
        meter->last_raw_reading[domain] = reading->joules_raw;
    }
    
    return 0;
}

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_StartMeasurement
 * Starts energy measurement (saves baseline)
 * ============================================================================ */
typedef struct {
    RIN_EnergyReading start_readings[RIN_RAPL_DOMAIN_MAX];
    uint64_t start_time_ns;
} RIN_EnergyMeasurement;

static inline int RIN_EnergyMeter_StartMeasurement(RIN_EnergyMeter* meter,
                                                    RIN_EnergyMeasurement* meas) {
    if (!meter || !meas) return -1;
    
    meas->start_time_ns = RIN_DPTM_GetTimestampNs();
    
    /* Read all available domains */
    for (int d = 0; d < RIN_RAPL_DOMAIN_MAX; d++) {
        RIN_EnergyMeter_ReadDomain(meter, d, &meas->start_readings[d]);
    }
    
    return 0;
}

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_EndMeasurement
 * Ends measurement and calculates delta
 * 
 * Returns: energy consumed in Joules (PKG domain by default)
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
 * FUNCTION: RIN_EnergyMeter_ComputeMetrics
 * Computes derived metrics
 * 
 * @joules_consumed: Total measured energy
 * @tokens_processed: Generated/processed tokens
 * @time_ns: Elapsed time in nanoseconds
 * @metrics: Structure to populate
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
        
        /* Equivalent power per 1000 tokens */
        double tokens_per_sec = (double)tokens_processed / time_seconds;
        if (tokens_per_sec > 0) {
            double seconds_per_1k = 1000.0 / tokens_per_sec;
            metrics->watts_per_1k = metrics->watts_average;  /* Same power, scaled */
        }
    }
}

/* ============================================================================
 * FUNCTION: RIN_EnergyMeter_Close
 * Cleanup
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
 * FUNCTION: RIN_EnergyMeter_PrintMetrics
 * Prints formatted metrics
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
 * FUNCTION: RIN_EnergyMeter_ValidateTarget
 * Validates if efficiency target is met
 * 
 * Target: < 0.5W per 1000 tokens
 * ============================================================================ */
static inline bool RIN_EnergyMeter_ValidateTarget(const RIN_EnergyMetrics* metrics,
                                                 float target_watts_per_1k) {
    if (!metrics) return false;
    
    return metrics->joules_per_1k_tokens <= target_watts_per_1k;
}

/* ============================================================================
 * MACRO: RIN_ENERGY_MEASURE_SCOPE
 * Macro to automatically measure a block of code
 * 
 * Usage:
 *   RIN_ENERGY_MEASURE_SCOPE(meter, metrics, {
 *       // ... code to measure ...
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
        /* Assumes tokens are counted externally */ \
        RIN_EnergyMeter_ComputeMetrics(_joules, 0, _time_ns, &(metrics_var)); \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* RIN_ENERGY_METER_H */
