# Must be the last Makefile included after all other modules
ifeq ("$(HCLIB_ROOT)", "")
  $(error Please set the HCLIB_ROOT environment variable.)
endif

HCLIB_CFLAGS=-I$(HCLIB_ROOT)/include
HCLIB_CXXFLAGS=-std=c++11 $(HCLIB_CFLAGS)
HCLIB_LDFLAGS=-L$(HCLIB_ROOT)/lib

ifeq ($(findstring hcos, $(CC)),hcos)
    IS_MAC_OS = 0
    HCLIB_CFLAGS+=-pthread
    HCLIB_CXXFLAGS+=-pthread
    HCLIB_LDFLAGS+=-pthread
    HCLIB_LDLIBS=-lhclib
else
UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
    IS_MAC_OS = 0
    HCLIB_LDLIBS=-lhclib -lrt -ldl
else
    IS_MAC_OS = 1
    HCLIB_LDLIBS=-lhclib
endif
endif

ifdef TBB_MALLOC
  HCLIB_LDFLAGS+=-L$(TBB_MALLOC)
  HCLIB_LDLIBS+=-ltbbmalloc_proxy
endif
