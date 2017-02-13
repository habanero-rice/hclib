include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=task_spawn_hclib task_spawn_omp task_wait_flat_hclib \
		task_wait_flat_omp task_wait_recursive_hclib task_wait_recursive_omp \
		future_spawn_hclib future_spawn_omp fan_out_hclib fan_out_omp \
		fan_out_and_in_hclib fan_out_and_in_omp bin_fan_out_hclib bin_fan_out_omp \
		parallel_loop_hclib parallel_loop_omp prod_cons_omp prod_cons_hclib \
		reduce_var_omp reduce_var_hclib unbalanced_bin_fan_out_omp \
		unbalanced_bin_fan_out_hclib

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

fan_out_and_in_omp: fan_out_and_in_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

bin_fan_out_omp: bin_fan_out_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

parallel_loop_omp: parallel_loop_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

prod_cons_omp: prod_cons_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

reduce_var_omp: reduce_var_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

unbalanced_bin_fan_out_omp: unbalanced_bin_fan_out_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

%: %.c
	$(CC) $(FLAGS) $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean:
	rm -f $(TARGETS)
