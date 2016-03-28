include $(HCLIB_ROOT)/include/hclib.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_cuda.o

TARGETS=lib/libhclib_cuda.so lib/libhclib_cuda.a

lib/libhclib_cuda.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_cuda.so: $(OBJS)
	g++ -shared $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) -o $@ $(OBJS)

obj/hclib_cuda.o: src/hclib_cuda.cpp
	g++ -std=c++11 -Iinc -I$(CUDA_HOME)/include $^ -o $@ $(PROJECT_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*
