include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/sos/inc/hclib_sos.post.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_sos.o

TARGETS=lib/libhclib_sos.so lib/libhclib_sos.a

all: $(TARGETS)

lib/libhclib_sos.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_sos.so: $(OBJS)
	g++ -shared -I../common $(HCLIB_LDFLAGS) $(HCLIB_SOS_LDFLAGS) $(HCLIB_LDLIBS) -lsma -o $@ $(OBJS)

obj/hclib_sos.o: src/hclib_sos.cpp
	g++ -std=c++11 -I../common -Iinc $^ -o $@ $(HCLIB_CFLAGS) $(HCLIB_SOS_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
