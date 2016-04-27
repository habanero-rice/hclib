include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.post.mak

TARGETS=samplesort_omp_shmem samplesort_mpi samplesort_hclib_shmem

FLAGS=-O3

all: $(TARGETS)

samplesort_omp_shmem: samplesort_omp_shmem.c
	oshcc $(FLAGS) samplesort_omp_shmem.c -o samplesort_omp_shmem -openmp

samplesort_mpi: samplesort_mpi.c
	mpiicc $(FLAGS) samplesort_mpi.c -o samplesort_mpi

samplesort_hclib_shmem: samplesort_hclib_shmem.cpp
	g++ -std=c++11 $(FLAGS) samplesort_hclib_shmem.cpp -o samplesort_hclib_shmem \
		$(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) $(HCLIB_LDFLAGS) \
		$(HCLIB_OSHMEM_LDFLAGS) $(HCLIB_LDLIBS) $(HCLIB_OSHMEM_LDLIBS)

clean:
	rm -f $(TARGETS)
