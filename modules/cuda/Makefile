include $(HCLIB_ROOT)/include/hclib.mak

DEFINES=-DHC_ASSERTION_CHECK
# DEFINES=-DVERBOSE

OBJS=obj/hclib_cuda.o

TARGETS=lib/libhclib_cuda.so lib/libhclib_cuda.a

all: $(TARGETS)

lib/libhclib_cuda.a: $(OBJS)
	ar rvs $@ $(OBJS)

lib/libhclib_cuda.so: $(OBJS)
	$(CXX) -shared -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o $@ $(OBJS)

obj/hclib_cuda.o: src/hclib_cuda.cpp
	$(CXX) -std=c++11 -Iinc -I$(CUDA_HOME)/include $^ -o $@ $(HCLIB_CFLAGS) --compile -fPIC $(DEFINES)

clean:
	rm -f obj/*.o lib/*

install: $(TARGETS)
	cp lib/*.a lib/*.so $(HCLIB_ROOT)/lib
	cp inc/*.h $(HCLIB_ROOT)/include/
	cp ../common/hclib-module-common.h $(HCLIB_ROOT)/include/
