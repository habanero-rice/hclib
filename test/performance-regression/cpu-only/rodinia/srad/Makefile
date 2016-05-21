include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak
# C compiler
HCLIB_PERF_CXX?=icpc
CC = $(HCLIB_PERF_CXX)
CC_FLAGS = -g -fopenmp -O2 $(HCLIB_CXXFLAGS)

srad:
	$(CC) $(CC_FLAGS) srad.cpp -o srad  $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

clean:
	rm -f srad
