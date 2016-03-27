include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/openshmem/inc/hclib_openshmem.post.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_openshmem.o

TARGETS=lib/libhclib_openshmem.so lib/libhclib_openshmem.a

lib/libhclib_openshmem.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_openshmem.so: $(OBJS)
	g++ -shared $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) -o $@ $(OBJS)

obj/hclib_openshmem.o: src/hclib_openshmem.cpp
	g++ -std=c++11 -Iinc $^ -o $@ $(PROJECT_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
