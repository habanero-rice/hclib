include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

TARGETS=jacobi-task jacobi-block-for jacobi-block-task
HCLIB_PERF_CXX?=icpc

all: $(TARGETS)

SHARED_SRC=main.c src/jacobi-seq.c src/poisson.c
FLAGS=-Iinclude -I../common $(HCLIB_CXXFLAGS)

jacobi-task: src/jacobi-task.c $(SHARED_SRC)
	rm -f *.o
	$(HCLIB_PERF_CXX) -c src/jacobi-task.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c main.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c src/jacobi-seq.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c src/poisson.c $(FLAGS)
	$(HCLIB_PERF_CXX) jacobi-task.o main.o jacobi-seq.o poisson.o $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o $@

jacobi-block-for: src/jacobi-block-for.c $(SHARED_SRC)
	rm -f *.o
	$(HCLIB_PERF_CXX) -c src/jacobi-block-for.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c main.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c src/jacobi-seq.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c src/poisson.c $(FLAGS)
	$(HCLIB_PERF_CXX) jacobi-block-for.o main.o jacobi-seq.o poisson.o $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o $@

jacobi-block-task: src/jacobi-block-task.c $(SHARED_SRC)
	rm -f *.o
	$(HCLIB_PERF_CXX) -c src/jacobi-block-task.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c main.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c src/jacobi-seq.c $(FLAGS)
	$(HCLIB_PERF_CXX) -c src/poisson.c $(FLAGS)
	$(HCLIB_PERF_CXX) jacobi-block-task.o main.o jacobi-seq.o poisson.o $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o $@

clean:
	rm -f $(TARGETS) *.o
