include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/sos/inc/hclib_sos.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/sos/inc/hclib_sos.post.mak

TARGETS=init many_putmem shmem_malloc shmem_barrier_all \
		shmem_broadcast64 shmem_int_async_when \
		shmem_int_wait_until shmem_lock_stress \
		shmem_put64 shmem_atomics_stress shmem_barrier_stress

all: $(TARGETS)

%: %.cpp
	g++ -std=c++11 -I../../common $(HCLIB_CFLAGS) $(HCLIB_SOS_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_SOS_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_SOS_LDLIBS)

%: %.c
	gcc -I../../common $(HCLIB_CFLAGS) $(HCLIB_SOS_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_SOS_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_SOS_LDLIBS)


clean:
	rm -f $(TARGETS)
