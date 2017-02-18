include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=task_spawn_hclib task_spawn_omp task_spawn_tbb task_spawn_ocr task_spawn_realm task_wait_flat_hclib \
		task_wait_flat_omp task_wait_flat_tbb task_wait_flat_ocr task_wait_flat_realm task_wait_recursive_hclib task_wait_recursive_omp task_wait_recursive_tbb task_wait_recursive_ocr task_wait_recursive_realm \
		future_spawn_hclib future_spawn_omp future_spawn_tbb future_spawn_ocr future_spawn_realm fan_out_hclib fan_out_omp \
		fan_out_and_in_hclib fan_out_and_in_omp bin_fan_out_hclib bin_fan_out_omp \
		parallel_loop_hclib parallel_loop_omp prod_cons_omp prod_cons_hclib \
		reduce_var_omp reduce_var_hclib unbalanced_bin_fan_out_omp \
		unbalanced_bin_fan_out_hclib

CC?=g++
CXX?=g++
FLAGS=-g -O3 -Wall -Werror

all: $(TARGETS)

%_tbb: %_tbb.cpp
	$(CC) $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -I$(TBBROOT)/include -ltbb

%_omp: %_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -ltbbmalloc

%_ocr: %_ocr.c
	$(CXX) $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -I$(OCR_INSTALL)/include -locr_x86 -L$(OCR_INSTALL)/lib

%_realm: %_realm.cpp
	$(CXX) $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) $(REALM_LIB) -I$(LG_RT_DIR)

%: %.c
	$(CC) $(FLAGS) $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) -ltbbmalloc

clean:
	rm -f $(TARGETS)
