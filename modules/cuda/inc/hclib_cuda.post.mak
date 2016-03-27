# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
PROJECT_CFLAGS+=-I$(HCLIB_ROOT)/../modules/cuda/inc
PROJECT_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/cuda/inc
PROJECT_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/cuda/lib -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64
PROJECT_LDLIBS += $(call GET_LINK_FLAG,--whole-archive) \
				  $(call GET_LINK_FLAG,-lhclib_cuda) \
				  $(call GET_LINK_FLAG,--no-whole-archive) \
				  $(call GET_LINK_FLAG,-lcudart)
