include $(HCLIB_ROOT)/include/hclib.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_mpi.o

TARGETS=lib/libhclib_mpi.so lib/libhclib_mpi.a

lib/libhclib_mpi.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_mpi.so: $(OBJS)
	g++ -shared $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) -o $@ $(OBJS)

obj/hclib_mpi.o: src/hclib_mpi.cpp
	g++ -std=c++11 -Iinc $^ -o $@ $(PROJECT_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
