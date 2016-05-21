include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

HCLIB_PERF_CXX?=icpc
CCFLAGS = -g -fopenmp -O3
OUTPUT = *.out

3D:
	$(HCLIB_PERF_CXX) -o 3D $(CCFLAGS) 3D.c -lm  $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

clean:
	rm -f 3D $(OUTPUT)
