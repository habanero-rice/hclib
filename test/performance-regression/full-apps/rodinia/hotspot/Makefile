include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak
# C compiler
HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)
CC_FLAGS = -g -O2

all: hotspot

hotspot: hotspot_openmp.o Makefile 
	$(CC) $(CC_FLAGS) hotspot.o -o hotspot $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

hotspot_openmp.o: hotspot_openmp.cpp Makefile
	$(CC) -c $(CC_FLAGS) hotspot_openmp.cpp -o hotspot.o $(HCLIB_CXXFLAGS)

clean:
	rm -f hotspot *.o
