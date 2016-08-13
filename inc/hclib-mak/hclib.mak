ifeq ("$(HCLIB_ROOT)", "")
  $(error Please set teh HCLIB_ROOT environment variable.)
endif

PROJECT_CFLAGS=-I$(HCLIB_ROOT)/include $(shell xml2-config --cflags)
PROJECT_CXXFLAGS=-std=c++11 $(PROJECT_CFLAGS)
PROJECT_LDFLAGS=-L$(HCLIB_ROOT)/lib
PROJECT_LDLIBS=-lhclib -Wl,-rpath,$(HCLIB_ROOT)/lib $(shell xml2-config --libs)
ifdef TBB_MALLOC
  PROJECT_LDFLAGS+=-L$(TBB_MALLOC)
  PROJECT_LDLIBS+=-ltbbmalloc_proxy
endif
ifdef HC_CUDA_FLAGS
  PROJECT_LDFLAGS+=-L$(CUDA_HOME)/lib
  PROJECT_LDFLAGS+=-L$(CUDA_HOME)/lib64
  PROJECT_CFLAGS+=-I$(CUDA_HOME)/include
  PROJECT_LDLIBS+=-lcudart
  PROJECT_CFLAGS+=-DHC_CUDA
  PROJECT_CXXFLAGS+=-DHC_CUDA
endif
