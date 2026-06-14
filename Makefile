CC = gcc
CFLAGS = -O3 -march=native -mtune=znver3 -mavx2 -mfma -fopenmp \
         -funroll-loops -flto -ffast-math -D_GNU_SOURCE -fPIC \
         -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
         -Wno-sign-compare -Wno-missing-field-initializers
LDFLAGS = -lm -lrt -fopenmp -flto
AR = ar rcs

# ============================================================================
# TARGETS
# ============================================================================
.PHONY: all clean benchmark test demo shared

all: rin_test rin_demo librin.so

# ============================================================================
# OBJECTS
# ============================================================================
COMMON_OBJS = rin_core.o

# ============================================================================
# RIN TEST (full module tests)
# ============================================================================
rin_test: rin_test.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

rin_test.o: rin_test.c rin_core.h
	$(CC) $(CFLAGS) -c -o $@ rin_test.c

# ============================================================================
# RIN DEMO (main demo)
# ============================================================================
rin_demo: main.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

main.o: main.c rin_core.h
	$(CC) $(CFLAGS) -c -o $@ main.c

# ============================================================================
# RIN CORE
# ============================================================================
rin_core.o: rin_core.c rin_core.h rin_arena.h rin_dptm.h rin_lif_engine.h \
            rin_ptsoftmax.h rin_bspn.h rin_dct_engine.h rin_phase_gating.h \
            rin_betti_calculator.h rin_mechanistic_distill.h \
            rin_energy_meter.h rin_test_suite.h
	$(CC) $(CFLAGS) -c -o $@ rin_core.c

# ============================================================================
# EXPERIMENTAL KERNELS (build separately, see experimental/rin-x/)
# ============================================================================
# rinx_ultra_int8 rinx_batch rinx_tiny rinx_extreme moved to experimental/

# ============================================================================
# THORIN SHARED LIBRARY (Python bindings)
# ============================================================================
thorin_api.o: thorin_api.c thorin_api.h rin_core.h
	$(CC) $(CFLAGS) -fPIC -c -o $@ thorin_api.c

librin.so: thorin_api.o $(COMMON_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

.PHONY: shared
shared: librin.so

# ============================================================================
# CLEAN
# ============================================================================
clean:
	rm -f *.o rin_test rin_demo librin.so

# ============================================================================
# SHORTCUTS
# ============================================================================
test: rin_test
	./rin_test

demo: rin_demo
	./rin_demo --benchmark
