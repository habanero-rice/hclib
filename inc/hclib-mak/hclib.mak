# Must be the last Makefile included after all other modules
PROJECT_CFLAGS=-I$(HCLIB_ROOT)/include -I$(LIBXML2_INCLUDE)
PROJECT_CXXFLAGS=-std=c++11 $(PROJECT_CFLAGS)
PROJECT_LDFLAGS=-L$(LIBXML2_LIBS) -L$(HCLIB_ROOT)/lib
PROJECT_LDLIBS=-lhclib -lxml2 $(JSMN_HOME)/libjsmn.a -lrt

ifndef GET_LINK_FLAG
	GET_LINK_FLAG = -Wl,$(1)
endif
