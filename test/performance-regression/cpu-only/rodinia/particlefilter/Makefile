include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
#makefile

HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)

particle_filter: ex_particle_OPENMP_seq.c
	$(CC) -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm  $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)


clean:
	rm particle_filter
