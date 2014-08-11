HCPPONLY = 1

include $(HCPP_INSTALL)/include/hcpp.mak

EXE=fib asyncAwait0 nqueens fib-ddt

all: clean $(EXE) clean-obj

fib: fib.cpp
	$(CXX) $(PROJECT_LDFLAGS) $(PROJECT_CXXFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

fib-ddt: fib-ddt.cpp
	$(CXX) $(PROJECT_LDFLAGS) $(PROJECT_CXXFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

asyncAwait0: asyncAwait0.cpp
	$(CXX) $(PROJECT_LDFLAGS) $(PROJECT_CXXFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

nqueens: nqueens.cpp
	$(CXX) $(PROJECT_LDFLAGS) $(PROJECT_CXXFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
