# Must be the last Makefile included after all other modules
PROJECT_CFLAGS=-I$(HCLIB_ROOT)/include -I$(LIBXML2_INCLUDE)
PROJECT_CXXFLAGS=-std=c++11 $(PROJECT_CFLAGS)
PROJECT_LDFLAGS=-L$(LIBXML2_LIBS) -L$(HCLIB_ROOT)/lib

ifndef GET_LINK_FLAG
	GET_LINK_FLAG = -Wl,$(1)
endif

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
	PROJECT_LDLIBS+=-lrt
	IS_MAC_OS = 0
	PROJECT_LDLIBS=-lhclib -lxml2 $(JSMN_HOME)/libjsmn.a
else
	IS_MAC_OS = 1
	PROJECT_LDLIBS=-lhclib -lxml2 $(call GET_LINK_FLAG,-force_load) \
				   $(call GET_LINK_FLAG,$(JSMN_HOME)/libjsmn.a)
endif
