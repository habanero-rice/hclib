include $(CRT_ROOT)/include/crt.mak

EXE=UTS

all: clean $(EXE) clean-obj

UTS: UTS.cpp
	g++ -g -Wno-write-strings $(PROJECT_CXXFLAGS) -DBRG_RNG $(PROJECT_LDFLAGS) -o $@ $^ uts.c rng/brg_sha1.c uts_dm_hooks.c $(PROJECT_LDLIBS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
