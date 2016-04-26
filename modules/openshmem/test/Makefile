include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.post.mak

TARGETS=init shmem_malloc samplesort_omp shmem_barrier_all shmem_put64 shmem_broadcast64 shmem_lock_stress

all: $(TARGETS)

%: %.cpp
	g++ -std=c++11 $(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_OSHMEM_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_OSHMEM_LDLIBS)

%: %.c
	gcc $(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_OSHMEM_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_OSHMEM_LDLIBS)


clean:
	rm -f $(TARGETS)
