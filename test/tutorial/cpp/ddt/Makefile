include $(HCLIB_ROOT)/include/hclib.mak

EXE=fib graph

all: clean $(EXE) clean-obj

graph: graph.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

fib: fib.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
