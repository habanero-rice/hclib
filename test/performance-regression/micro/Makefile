include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=task_spawn_rate_hclib task_spawn_rate_omp

FLAGS=-g

all: $(TARGETS)

task_spawn_rate_omp: task_spawn_rate_omp.c
	g++ -O3 -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

%: %.c
	g++ -O3 $(FLAGS) $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean:
	rm -f $(TARGETS)
