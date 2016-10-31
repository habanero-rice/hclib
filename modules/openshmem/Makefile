include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.post.mak

DEFINES=-DHC_ASSERTION_CHECK -DPROFILE
# DEFINES=-DVERBOSE

OBJS=obj/hclib_openshmem.o

TARGETS=lib/libhclib_openshmem.so lib/libhclib_openshmem.a

all: $(TARGETS)

lib/libhclib_openshmem.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_openshmem.so: $(OBJS)
	g++ -shared $(HCLIB_LDFLAGS) $(HCLIB_OSHMEM_LDFLAGS) $(HCLIB_LDLIBS) -o $@ $(OBJS)

obj/hclib_openshmem.o: src/hclib_openshmem.cpp
	g++ -std=c++11 -Iinc $^ -o $@ $(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
