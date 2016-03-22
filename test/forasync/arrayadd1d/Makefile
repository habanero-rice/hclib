include $(HCPP_ROOT)/include/hcpp.mak

EXE=arrayadd1d

all: clean $(EXE) clean-obj

arrayadd1d: arrayadd1d.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
