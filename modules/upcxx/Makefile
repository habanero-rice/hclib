include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/upcxx/inc/hclib_upcxx.post.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_upcxx.o

TARGETS=lib/libhclib_upcxx.a

all: $(TARGETS)

lib/libhclib_upcxx.a: $(OBJS)
	ar rvs $@ $(OBJS)

obj/hclib_upcxx.o: src/hclib_upcxx.cpp
	/opt/gcc/4.9.0/bin/gcc -std=c++11 -Iinc $^ -o $@ $(HCLIB_CFLAGS) $(HCLIB_UPCXX_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
