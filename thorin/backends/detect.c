/* THORIN backend auto-detection — thin wrapper that picks the best available SIMD
 * backend at runtime.
 */
#include "thorin_backend.h"

const char* thor_best_backend(void) {
#if defined(__x86_64__) || defined(__i386__)
    /* x86 auto-detection is done at the Python level via /proc/cpuinfo */
    return "x86";
#elif THOR_HAVE_NEON
    return "neon";
#elif THOR_HAVE_WASM_SIMD
    return "wasm";
#else
    return "generic";
#endif
}
