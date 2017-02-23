include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TBB_MALLOC_LIBDIR?=/opt/apps/intel/psxe_2017_u1/tbb/lib/intel64/gcc4.7

HCLIB_TARGETS=task_spawn_hclib task_wait_flat_hclib future_spawn_hclib fan_out_hclib \
	fan_out_and_in_hclib bin_fan_out_hclib parallel_loop_hclib prod_cons_hclib \
	reduce_var_hclib unbalanced_bin_fan_out_hclib
OMP_TARGETS=task_spawn_omp task_wait_flat_omp future_spawn_omp fan_out_omp \
	fan_out_and_in_omp bin_fan_out_omp parallel_loop_omp prod_cons_omp \
	reduce_var_omp unbalanced_bin_fan_out_omp
TBB_TARGETS=task_spawn_tbb task_wait_flat_tbb future_spawn_tbb fan_out_tbb \
	fan_out_and_in_tbb bin_fan_out_tbb parallel_loop_tbb prod_cons_tbb \
	unbalanced_bin_fan_out_tbb
OCR_TARGETS=task_spawn_ocr task_wait_flat_ocr future_spawn_ocr fan_out_ocr \
	fan_out_and_in_ocr bin_fan_out_ocr parallel_loop_ocr prod_cons_ocr \
	reduce_var_ocr unbalanced_bin_fan_out_ocr
REALM_TARGETS=task_spawn_realm task_wait_flat_realm future_spawn_realm fan_out_realm \
	fan_out_and_in_realm bin_fan_out_realm parallel_loop_realm prod_cons_realm \
	unbalanced_bin_fan_out_realm

CC?=g++
CXX?=g++
FLAGS=-g -O3 -Wall -Werror

all: hclib omp tbb ocr realm

hclib: $(HCLIB_TARGETS)
omp: $(OMP_TARGETS)
tbb: $(TBB_TARGETS)
ocr: $(OCR_TARGETS)
realm: $(REALM_TARGETS)

%_tbb: %_tbb.cpp
	$(CC) -std=c++11 $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -I$(TBBROOT)/include -ltbb -L$(TBBROOT)/lib/intel64/gcc4.7

%_omp: %_omp.c
	$(CC) -fopenmp $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -L$(TBB_MALLOC_LIBDIR) -ltbbmalloc

%_ocr: %_ocr.c
	$(CXX) $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -I$(OCR_INSTALL)/include -locr_x86 -L$(OCR_INSTALL)/lib

%_realm: %_realm.cpp
	$(CXX) $(FLAGS) $(HCLIB_CFLAGS) -o $@ $^ $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) $(REALM_LIB) -I$(LG_RT_DIR)

%: %.c
	$(CC) $(FLAGS) $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) -L$(TBB_MALLOC_LIBDIR) -ltbbmalloc

clean:
	rm -f $(HCLIB_TARGETS) $(OMP_TARGETS) $(TBB_TARGETS) $(OCR_TARGETS) $(REALM_TARGETS)
