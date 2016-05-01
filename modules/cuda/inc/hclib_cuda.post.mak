# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
HCLIB_CUDA_CFLAGS+=-I$(HCLIB_ROOT)/../modules/cuda/inc
HCLIB_CUDA_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/cuda/inc
HCLIB_CUDA_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/cuda/lib -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64
HCLIB_CUDA_LDLIBS+=-lhclib_cuda -lcudart
