include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/cuda/inc/hclib_cuda.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/cuda/inc/hclib_cuda.post.mak

TARGETS=init allocate kernel

all: $(TARGETS)

%: %.cu
	nvcc -arch=sm_37 -std=c++11 $(HCLIB_CFLAGS) $(HCLIB_CUDA_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_CUDA_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_CUDA_LDLIBS)

clean:
	rm -f $(TARGETS)
