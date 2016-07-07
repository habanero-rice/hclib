include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)
LOCAL_CC = $(CC) -g -O3
CFLAGS = -lm $(HCLIB_CXXFLAGS)


all : nn

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_openmp.c
	$(CC) -c -o nn_openmp.o $< $(LDFLAGS) $(CFLAGS)
	$(CC) -o $@ nn_openmp.o $(LDFLAGS) -lm $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

