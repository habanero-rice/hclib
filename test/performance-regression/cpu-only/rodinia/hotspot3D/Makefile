include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
CC = gcc
CCFLAGS = -g -fopenmp -O3
OUTPUT = *.out

3D:
	$(CC) -o 3D $(CCFLAGS) 3D.c -lm  $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

clean:
	rm -f 3D $(OUTPUT)
