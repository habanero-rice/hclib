include $(HCLIB_ROOT)/include/hclib.mak

EXE=matmul averaging arrayadd1d

all: clean $(EXE) clean-obj

arrayadd1d: arrayadd1d.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

averaging: averaging.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

matmul: matmul.cpp
	$(CXX) -O3 $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
