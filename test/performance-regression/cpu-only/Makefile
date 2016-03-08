include $(HCLIB_ROOT)/include/hclib.mak

TARGETS=cilksort FFT fib fib-ddt nqueens qsort

all: ${TARGETS}

%: %.cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -rf *.o ${TARGETS} *.dSYM
