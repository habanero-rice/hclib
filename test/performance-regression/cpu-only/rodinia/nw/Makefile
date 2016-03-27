include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -O3 -fopenmp $(PROJECT_CXXFLAGS)

all: needle

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle  $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

clean:
	rm -f needle
