include $(HCPP_ROOT)/include/hcpp.mak

EXE=hpt

all: clean $(EXE) clean-obj

hpt: asyncAtPlace0.cpp
	g++ $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
