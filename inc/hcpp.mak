PROJECT_CXXFLAGS=-std=c++11 -I$(HCLIB_ROOT)/include -I$(HCPP_ROOT)/include
PROJECT_LDFLAGS=
PROJECT_LDLIBS=
ifndef TBB_MALLOC
  PROJECT_LDFLAGS=-L$(HCLIB_ROOT)/lib -L$(OCR_ROOT)/lib  -L$(HCPP_ROOT)/lib
  PROJECT_LDLIBS=-lhclib -locr -lhcpp
else
  PROJECT_LDFLAGS=-L$(HCLIB_ROOT)/lib -L$(OCR_ROOT)/lib  -L$(HCPP_ROOT)/lib -L$(TBB_MALLOC)
  PROJECT_LDLIBS=-lhclib -locr -lhcpp -ltbbmalloc_proxy
endif
