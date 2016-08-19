include $(HCLIB_ROOT)/include/hclib.mak

EXE=UTS

all: clean $(EXE) clean-obj

UTS: UTS.cpp
	$(CXX) -Wno-write-strings $(PROJECT_CXXFLAGS) -DBRG_RNG $(PROJECT_LDFLAGS) -o $@ $^ uts.c rng/brg_sha1.c $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
