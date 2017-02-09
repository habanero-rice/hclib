include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=task_spawn_hclib task_spawn_omp task_wait_flat_hclib \
		task_wait_flat_omp task_wait_recursive_hclib task_wait_recursive_omp \
		future_spawn_hclib future_spawn_omp fan_out_hclib fan_out_omp

CC?=g++
FLAGS=-g -O3

all: $(TARGETS)

task_spawn_omp: task_spawn_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

task_wait_flat_omp: task_wait_flat_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

task_wait_recursive_omp: task_wait_recursive_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

future_spawn_omp: future_spawn_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

fan_out_omp: fan_out_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

%: %.c
	$(CC) $(FLAGS) $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean:
	rm -f $(TARGETS)
