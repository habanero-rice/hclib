include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=cilksort FFT fib fib-ddt nqueens qsort
HCLIB_PERF_CXX?=icpc

all: ${TARGETS}

%: %.cpp
	$(HCLIB_PERF_CXX) $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean:
	rm -rf *.o ${TARGETS} *.dSYM
