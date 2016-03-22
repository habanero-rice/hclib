include $(HCPP_ROOT)/include/hcpp.mak

EXE=smith_waterman

all: clean $(EXE) clean-obj

smith_waterman: smith_waterman.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM *.out
