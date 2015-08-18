PROJECT_CXXFLAGS=-std=c++11 -I$(HCPP_ROOT)/include -I$(LIBXML2)/include/libxml2
PROJECT_LDFLAGS=-L$(LIBXML2)/lib -L$(HCPP_ROOT)/lib
PROJECT_LDLIBS=-lxml2 -lhcpp
ifdef TBB_MALLOC
  PROJECT_LDFLAGS+=-L$(TBB_MALLOC)
  PROJECT_LDLIBS+=-ltbbmalloc_proxy
endif
