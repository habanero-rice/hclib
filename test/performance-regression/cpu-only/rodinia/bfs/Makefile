include $(HCLIB_ROOT)/include/hclib.mak
# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2 $(PROJECT_CXXFLAGS)

all: bfs

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) bfs.cpp -o bfs

clean:
	rm -f bfs result.txt
