CC = gcc
VPATH = lib/src:lib/include:lib/include/experimental
CFLAGS = -O3 -march=native -mtune=znver3 -mavx2 -mfma -fopenmp \
         -funroll-loops -flto -ffast-math -D_GNU_SOURCE -fPIC \
         -I lib/include -I lib/include/experimental -I . \
         -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
         -Wno-sign-compare -Wno-missing-field-initializers
LDFLAGS = -lm -lrt -fopenmp -flto

.PHONY: all clean test demo shared

all: rin_test rin_demo librin.so

COMMON_OBJS = rin_core.o

rin_test: rin_test.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

rin_test.o: rin_test.c rin_core.h rin_test_suite.h
	$(CC) $(CFLAGS) -c -o $@ $<

rin_demo: main.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

main.o: main.c rin_core.h
	$(CC) $(CFLAGS) -c -o $@ $<

rin_core.o: rin_core.c rin_core.h rin_arena.h rin_dptm.h rin_lif_engine.h \
            rin_ptsoftmax.h rin_bspn.h rin_dct_engine.h rin_phase_gating.h \
            rin_betti_calculator.h rin_mechanistic_distill.h \
            rin_energy_meter.h rin_test_suite.h
	$(CC) $(CFLAGS) -c -o $@ $<

rin_api.o: rin_api.c rin_api.h rin_core.h
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

librin.so: rin_api.o $(COMMON_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

shared: librin.so

test: rin_test
	./rin_test

demo: rin_demo
	./rin_demo --benchmark

clean:
	rm -f *.o rin_test rin_demo librin.so
