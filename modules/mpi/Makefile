include $(HCLIB_ROOT)/include/hclib.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_mpi.o

TARGETS=lib/libhclib_mpi.so lib/libhclib_mpi.a

all: $(TARGETS)

lib/libhclib_mpi.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_mpi.so: $(OBJS)
	g++ -shared $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o $@ $(OBJS)

obj/hclib_mpi.o: src/hclib_mpi.cpp
	g++ -std=c++11 -Iinc -I${MPI_HOME}/include $^ -o $@ $(HCLIB_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
