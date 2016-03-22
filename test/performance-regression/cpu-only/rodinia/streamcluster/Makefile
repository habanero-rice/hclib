include $(HCLIB_ROOT)/include/hclib.mak
PREFIX=${PARSECDIR}/pkgs/kernels/streamcluster/inst/${PARSECPLAT}

TARGET_O = sc_omp

ifdef version
  ifeq "$(version)" "parallel"
    CXXFLAGS :=	$(CXXFLAGS) -DENABLE_THREADS -pthread
  endif
endif

all: omp

omp:
	g++ -O3 -fopenmp -o $(TARGET_O) streamcluster_omp.cpp $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)


clean:
	rm -f *.o *~ *.txt sc_cpu sc_omp
	
