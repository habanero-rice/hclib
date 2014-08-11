PROJECT_CXXFLAGS=-std=c++11 -I$(HCLIB_ROOT)/include -I$(HCPP_ROOT)/include
PROJECT_LDFLAGS=-L$(HCLIB_ROOT)/lib -L$(OCR_ROOT)/lib  -L$(HCPP_ROOT)/lib
PROJECT_LDLIBS=-lhclib -locr -lhcpp
