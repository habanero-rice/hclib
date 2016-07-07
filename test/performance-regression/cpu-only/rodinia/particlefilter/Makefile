include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak
#makefile

HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)

particle_filter: ex_particle_OPENMP_seq.c
	$(CC) -c -O3 ex_particle_OPENMP_seq.c -o particle_filter.o $(HCLIB_CXXFLAGS)
	$(CC) -O3 particle_filter.o -o particle_filter -lm  $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)


clean:
	rm -f particle_filter *.o
