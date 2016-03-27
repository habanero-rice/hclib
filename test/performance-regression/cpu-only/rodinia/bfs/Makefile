include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2 $(PROJECT_CXXFLAGS)

all: bfs

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) -o bfs

clean:
	rm -f bfs result.txt
