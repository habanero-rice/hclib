include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
#makefile

particle_filter: ex_particle_OPENMP_seq.c
	g++ -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm  $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)


clean:
	rm particle_filter
