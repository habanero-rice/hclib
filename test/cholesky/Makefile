include $(HCLIB_ROOT)/include/hclib.mak

EXE=cholesky

all: clean $(EXE) clean-obj

cholesky: cholesky.cpp sequential_cholesky.cpp trisolve.cpp update_diagonal.cpp update_nondiagonal.cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM *.out
