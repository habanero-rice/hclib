include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)
SRC = pathfinder.cpp
EXE = pathfinder
FLAGS = $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -g

release:
	$(CC) -c $(SRC) -o $(SRC).o -g $(HCLIB_CXXFLAGS)
	$(CC) $(SRC).o $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -g -o $(EXE)

clean:
	rm -f pathfinder *.o


