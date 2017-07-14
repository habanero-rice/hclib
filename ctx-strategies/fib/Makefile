BASE_TARGET := fib
EXTRA_TARGETS := cxx_fib omp_fib cilk_fib hpx_fib

include ../common/common.mk

cilk_fib: cilk_fib.c
	$(CC) -fcilkplus $(COMMON_FLAGS) $< -o $@

omp_fib: omp_fib.c
	$(CC) -fopenmp $(COMMON_FLAGS) $< -o $@

hpx_fib: hpx_fib.c
	$(CC) `pkg-config --cflags hpx` $< -o $@ `pkg-config --libs hpx`

