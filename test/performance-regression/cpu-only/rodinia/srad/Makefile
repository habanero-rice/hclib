include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O2 $(PROJECT_CXXFLAGS)

srad:
	$(CC) $(CC_FLAGS) srad.cpp -o srad  $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

clean:
	rm -f srad
