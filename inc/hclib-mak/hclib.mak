# Must be the last Makefile included after all other modules
HCLIB_CFLAGS=-I$(HCLIB_ROOT)/include -I$(LIBXML2_INCLUDE)
HCLIB_CXXFLAGS=-std=c++11 $(HCLIB_CFLAGS)
HCLIB_LDFLAGS=-L$(LIBXML2_LIBS) -L$(HCLIB_ROOT)/lib

ifndef GET_LINK_FLAG
	GET_LINK_FLAG = -Wl,$(1)
endif

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
	HCLIB_LDLIBS+=-lrt
	IS_MAC_OS = 0
	HCLIB_LDLIBS=-lhclib -lxml2 $(JSMN_HOME)/libjsmn.a -lrt
else
	IS_MAC_OS = 1
	HCLIB_LDLIBS=-lhclib -lxml2 $(call GET_LINK_FLAG,-force_load) \
				   $(call GET_LINK_FLAG,$(JSMN_HOME)/libjsmn.a)
endif
