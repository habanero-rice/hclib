# default to release build
PROJECT_GOALS := release

ifndef BASE_TARGET
BASE_TARGET := $(shell basename $$PWD)
endif

include $(HCLIB_ROOT)/include/hclib.mak

ALL_TARGETS := $(BASE_TARGET) nb_$(BASE_TARGET) hclang_$(BASE_TARGET)

COMMON_FLAGS := -O3

.PHONY: all
all: $(ALL_TARGETS)

CC_CMD  = $(CC) $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(PROJECT_LDLIBS)
CXX_CMD = $(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(PROJECT_LDLIBS)

%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/default
%: %.c $(PROJECT_EXTRA_DEPS)
	$(CC_CMD)

%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/default
%: %.cpp $(PROJECT_EXTRA_DEPS)
	echo $< $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

nb_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
nb_%: nb_%.c $(PROJECT_EXTRA_DEPS)
	$(CC_CMD)

nb_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
nb_%: nb_%.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

hclang_%: %.hc $(PROJECT_EXTRA_DEPS)
	hcc $(COMMON_FLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf *.o $(ALL_TARGETS) *.dSYM *.out rose_*.c
