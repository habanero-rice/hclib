include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/openshmem-am/inc/hclib_openshmem-am.post.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_openshmem-am.o

TARGETS=lib/libhclib_openshmem-am.a

all: $(TARGETS)

lib/libhclib_openshmem-am.a: $(OBJS)
	ar rvs $@ $(OBJS)

obj/hclib_openshmem-am.o: src/hclib_openshmem-am.cpp
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c++11 -fPIC -Iinc $^ -o $@ $(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) -c -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*

install: $(TARGETS)
	cp lib/*.a lib/*.so $(HCLIB_ROOT)/lib
	cp inc/*.h $(HCLIB_ROOT)/include/
