BASE_TARGET := fib
EXTRA_TARGETS := cxx_fib omp_fib cilk_fib hpx_fib
EXTRA_TARGETS += heap_args_fib ss4k_fib escaping_fib

include ../common/common.mk

cilk_fib: cilk_fib.c
	$(CC) -fcilkplus $(COMMON_FLAGS) $< -o $@

omp_fib: omp_fib.c
	$(CC) -fopenmp $(COMMON_FLAGS) $< -o $@

hpx_fib: hpx_fib.c
	$(CC) `pkg-config --cflags hpx` $< -o $@ `pkg-config --libs hpx`


# overhead variants

heap_args_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
heap_args_%: heap_args_%.c $(PROJECT_EXTRA_DEPS)
	$(CC_CMD)

cxx_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
cxx_%: cxx_%.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

ss4k_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/small-stacks
ss4k_%: fib.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

escaping_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
