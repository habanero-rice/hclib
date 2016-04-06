include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=jacobi-task jacobi-block-for jacobi-block-task-dep jacobi-block-task jacobi-task-dep

all: $(TARGETS)

SHARED_SRC=main.c src/jacobi-seq.c src/poisson.c
FLAGS=-Iinclude -I../common -fopenmp $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

jacobi-task: src/jacobi-task.c $(SHARED_SRC)
	g++ $^ $(FLAGS) -o $@

jacobi-block-for: src/jacobi-block-for.c $(SHARED_SRC)
	g++ $^ $(FLAGS) -o $@

jacobi-block-task-dep: src/jacobi-block-task-dep.c $(SHARED_SRC)
	g++ $^ $(FLAGS) -o $@

jacobi-block-task: src/jacobi-block-task.c $(SHARED_SRC)
	g++ $^ $(FLAGS) -o $@

jacobi-task-dep: src/jacobi-task-dep.c $(SHARED_SRC)
	g++ $^ $(FLAGS) -o $@

clean:
	rm -f $(TARGETS)
