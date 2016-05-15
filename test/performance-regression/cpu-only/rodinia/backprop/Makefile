include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

HCLIB_PERF_CC?=icc
CC = $(HCLIB_PERF_CC)

# C compiler
CC_FLAGS = -g -fopenmp -O2 $(HCLIB_CXXFLAGS)


backprop: backprop.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CC_FLAGS) backprop.o facetrain.o \
		imagenet.o backprop_kernel.o $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS) -o backprop -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
