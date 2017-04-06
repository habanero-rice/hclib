include $(HCLIB_ROOT)/include/hclib.mak

TARGETS := FFT Cilksort

all: clean $(TARGETS) clean-obj

%: %.cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(TARGETS) *.dSYM
