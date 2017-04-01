include $(HCLIB_ROOT)/include/hclib.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_system.o

TARGETS=lib/libhclib_system.so lib/libhclib_system.a

all: $(TARGETS)

lib/libhclib_system.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_system.so: $(OBJS)
	$(CXX) -shared -fPIC $(HCLIB_LDLIBS) $(HCLIB_LDFLAGS) -o $@ $(OBJS)

obj/hclib_system.o: src/hclib_system.cpp
	$(CXX) -c -fPIC -std=c++11 -Iinc $^ -o $@ $(HCLIB_CFLAGS) $(DEFINES)

clean:
	rm -f obj/*.o lib/*
