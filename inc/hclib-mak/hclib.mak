# Must be the last Makefile included after all other modules
ifeq ("$(HCLIB_ROOT)", "")
  $(error Please set the HCLIB_ROOT environment variable.)
endif

HCLIB_CFLAGS=-I$(HCLIB_ROOT)/include $(shell xml2-config --cflags)
HCLIB_CXXFLAGS=-std=c++11 $(HCLIB_CFLAGS)
HCLIB_LDFLAGS=-L$(HCLIB_ROOT)/lib

ifndef GET_LINK_FLAG
	GET_LINK_FLAG = -Wl,$(1)
endif

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
    IS_MAC_OS = 0
    HCLIB_LDLIBS=-lhclib $(shell xml2-config --libs) $(JSMN_HOME)/libjsmn.a -lrt -ldl
else
    IS_MAC_OS = 1
    HCLIB_LDLIBS=-lhclib $(shell xml2-config --libs) $(call GET_LINK_FLAG,-force_load) \
				   $(call GET_LINK_FLAG,$(JSMN_HOME)/libjsmn.a)
endif

ifdef TBB_MALLOC
  HCLIB_LDFLAGS+=-L$(TBB_MALLOC)
  HCLIB_LDLIBS+=-ltbbmalloc_proxy
endif
