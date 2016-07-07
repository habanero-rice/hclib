HCLIB_PERF_CC=nvcc
HCLIB_PERF_CXX=nvcc
HCLIB_CFLAGS+=-DOMP_TO_HCLIB_ENABLE_GPU -x cu -Xcompiler -fopenmp -I$(HCLIB_HOME)/modules/cuda/inc -DHCLIB_FORASYNC_MODE=FORASYNC_MODE_FLAT
HCLIB_LDFLAGS+=-Xcompiler -fopenmp -L$(HCLIB_HOME)/modules/cuda/lib
HCLIB_LDLIBS+=$(HCLIB_HOME)/modules/cuda/lib/libhclib_cuda.a
