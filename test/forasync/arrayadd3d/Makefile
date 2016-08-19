include $(HCLIB_ROOT)/include/hclib.mak

EXE=arrayadd3d

all: clean $(EXE) clean-obj

arrayadd3d: arrayadd3d.cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
