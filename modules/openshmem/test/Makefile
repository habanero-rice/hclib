include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.post.mak

TARGETS=init shmem_malloc shmem_barrier_all shmem_put64 \
		shmem_broadcast64 shmem_lock_stress shmem_int_wait_until \
		shmem_int_wait_until_any shmem_int_async_when shmem_int_async_when_any

all: $(TARGETS)

%: %.cpp
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c++11 $^ -I$(HCLIB_ROOT)/include \
		-I$(HCLIB_ROOT)/../modules/openshmem/inc -I$(HCLIB_ROOT)/../modules/system/inc \
		../lib/libhclib_openshmem.a -lhclib -L$(HCLIB_ROOT)/lib $(JSMN_HOME)/libjsmn.a -o $@
	# g++ -std=c++11 $(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_OSHMEM_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_OSHMEM_LDLIBS)

clean:
	rm -f $(TARGETS)
