include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

TARGETS=jacobi-task jacobi-block-for jacobi-block-task
HCLIB_PERF_CXX?=icpc

all: $(TARGETS)

SHARED_SRC=main.c src/jacobi-seq.c src/poisson.c
FLAGS=-Iinclude -I../common -fopenmp $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

jacobi-task: src/jacobi-task.c $(SHARED_SRC)
	$(HCLIB_PERF_CXX) $^ $(FLAGS) -o $@

jacobi-block-for: src/jacobi-block-for.c $(SHARED_SRC)
	$(HCLIB_PERF_CXX) $^ $(FLAGS) -o $@

jacobi-block-task: src/jacobi-block-task.c $(SHARED_SRC)
	$(HCLIB_PERF_CXX) $^ $(FLAGS) -o $@

clean:
	rm -f $(TARGETS)
