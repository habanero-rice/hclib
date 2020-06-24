include $(HCLIB_ROOT)/include/hclib.mak

EXE=cholesky

all: clean $(EXE) clean-obj

cholesky: cholesky.cpp sequential_cholesky.cpp trisolve.cpp update_diagonal.cpp update_nondiagonal.cpp
	$(CXX) ${FLAGS} $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM *.out
