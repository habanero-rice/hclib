ifeq ("$(HCLIB_ROOT)", "")
  $(error Please set teh HCLIB_ROOT environment variable.)
endif

PROJECT_BASE_FLAGS := -I$(HCLIB_ROOT)/include $(shell xml2-config --cflags)
PROJECT_CFLAGS     := -std=c11 $(PROJECT_BASE_FLAGS)
PROJECT_CXXFLAGS   := -std=c++11 $(PROJECT_BASE_FLAGS)
PROJECT_LDFLAGS    := -L$(HCLIB_ROOT)/lib
PROJECT_LDLIBS     := -lhclib -Wl,-rpath,$(HCLIB_ROOT)/lib $(shell xml2-config --libs)
ifdef TBB_MALLOC
  PROJECT_LDFLAGS+=-L$(TBB_MALLOC)
  PROJECT_LDLIBS+=-ltbbmalloc_proxy
endif
