include $(HCLIB_ROOT)/include/hclib.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_system.o

TARGETS=lib/libhclib_system.so lib/libhclib_system.a

all: $(TARGETS)

lib/libhclib_system.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_system.so: $(OBJS)
	g++ -shared $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) -o $@ $(OBJS)

obj/hclib_system.o: src/hclib_system.cpp
	g++ -std=c++11 -Iinc $^ -o $@ $(PROJECT_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
