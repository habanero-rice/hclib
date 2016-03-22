include $(HCLIB_ROOT)/include/hclib.mak
#makefile

particle_filter: ex_particle_OPENMP_seq.c
	gcc -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm  $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)


clean:
	rm particle_filter
