include $(HCPP_ROOT)/include/hcpp.mak

EXE=fib nqueens qsort fib-ddt asyncAwait0

all: clean $(EXE) clean-obj

asyncAwait0: asyncAwait0.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

nqueens: nqueens.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

qsort: qsort.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

fib-ddt: fib-ddt.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

fib: fib.cpp
	g++ $(PROJECT_CXXFLAGS) -o $@ $^ -L . $(PROJECT_LDLIBS) $(PROJECT_LDFLAGS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
