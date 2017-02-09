include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

HCLIB_PERF_CXX?=icpc
CC=$(HCLIB_PERF_CXX)

CC_FLAGS=-g -O2 $(HCLIB_CXXFLAGS)

all: bfs

bfs: bfs.o
	$(CC) -g -O2 -o bfs bfs.o $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

bfs.o: bfs.cpp
	$(CC) -c $(CC_FLAGS) bfs.cpp -o bfs.o

clean:
	rm -f bfs result.txt *.o
