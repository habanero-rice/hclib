include $(HCPP_ROOT)/include/hcpp.mak

EXE=arrayadd2d

all: clean $(EXE) clean-obj

arrayadd2d: arrayadd2d.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
