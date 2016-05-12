include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2 $(HCLIB_CXXFLAGS)

all: bfs

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o bfs

clean:
	rm -f bfs result.txt
