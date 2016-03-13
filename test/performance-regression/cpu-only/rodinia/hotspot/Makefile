include $(HCLIB_ROOT)/include/hclib.mak
# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2

all: hotspot

hotspot: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

clean:
	rm -f hotspot
