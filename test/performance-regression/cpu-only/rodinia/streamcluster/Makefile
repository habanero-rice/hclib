include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
PREFIX=${PARSECDIR}/pkgs/kernels/streamcluster/inst/${PARSECPLAT}

TARGET_O = sc_omp

ifdef version
  ifeq "$(version)" "parallel"
    CXXFLAGS :=	$(CXXFLAGS) -DENABLE_THREADS -pthread
  endif
endif

HCLIB_PERF_CXX?=icpc

all: omp

omp:
	$(HCLIB_PERF_CXX) -O3 -fopenmp -o $(TARGET_O) streamcluster_omp.cpp $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)


clean:
	rm -f *.o *~ *.txt sc_cpu sc_omp
	
