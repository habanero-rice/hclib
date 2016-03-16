include $(HCLIB_ROOT)/include/hclib.mak
LOCAL_CC = gcc -g -O3 -Wall
CC = gcc
CFLAGS = -lm -fopenmp -Wall $(PROJECT_CXXFLAGS)


all : nn

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

