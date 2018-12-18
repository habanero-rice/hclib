include $(HCLIB_ROOT)/include/hclib.mak

EXE=arraysum fib

all: clean $(EXE) clean-obj

fib: fib.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

arraysum: arraysum.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
