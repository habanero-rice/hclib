include $(HCLIB_ROOT)/include/hclib.mak

EXE=hpt

all: clean $(EXE) clean-obj

hpt: asyncAtPlace0.cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
