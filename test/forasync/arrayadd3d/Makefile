include $(HCPP_ROOT)/include/hcpp.mak

EXE=arrayadd3d

all: clean $(EXE) clean-obj

arrayadd3d: arrayadd3d.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
