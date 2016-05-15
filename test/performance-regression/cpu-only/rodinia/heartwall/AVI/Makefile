include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)

all: avimod.o avilib.o

# compile supporting function file into object (binary)
avimod.o: avilib.h avimod.c 
	$(CC) -c avimod.c $(HCLIB_CXXFLAGS)

# compile supporting function file into object (binary)
avilib.o: avilib.h avilib.c
	$(CC) -c avilib.c $(HCLIB_CXXFLAGS)

# delete files
clean:
	rm *.o *.out
