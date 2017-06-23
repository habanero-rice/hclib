# default to release build
PROJECT_GOALS := release

ifndef BASE_TARGET
BASE_TARGET := $(shell basename $$PWD)
endif

include $(HCLIB_ROOT)/include/hclib.mak

ALL_TARGETS := $(BASE_TARGET) nb_$(BASE_TARGET) hclang_$(BASE_TARGET) $(EXTRA_TARGETS)

COMMON_FLAGS := -O3

.PHONY: all
all: $(ALL_TARGETS)

CC_CMD  = $(CC) $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) -o $@ $^ $(PROJECT_LDLIBS)
CXX_CMD = $(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) -o $@ $^ $(PROJECT_LDLIBS)

%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/default
%: %.c
	$(CC_CMD)

%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/default
%: %.cpp
	$(CXX_CMD)

nb_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
nb_%: nb_%.c
	$(CC_CMD)

nb_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
nb_%: nb_%.cpp
	$(CXX_CMD)

hclang_%: %.hc
	hcc $(COMMON_FLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -rf *.o $(ALL_TARGETS) *.dSYM *.out rose_*.c
