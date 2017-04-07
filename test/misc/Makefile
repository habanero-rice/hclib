include $(HCLIB_ROOT)/include/hclib.mak

TARGETS := fib nqueens qsort fib-ddt

all: clean $(TARGETS) clean-obj

%: %.cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(TARGETS) *.dSYM
