PROJECT_CFLAGS+=-I$(HCLIB_ROOT)/../modules/cuda/inc
PROJECT_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/cuda/inc
PROJECT_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/cuda/lib -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64
PROJECT_LDLIBS+=-Wl,--whole-archive -lhclib_cuda -Wl,--no-whole-archive -lcudart
