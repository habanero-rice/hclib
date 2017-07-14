# default to release build
PROJECT_GOALS := release

ifndef BASE_TARGET
BASE_TARGET := $(shell basename $$PWD)
endif

include $(HCLIB_ROOT)/include/hclib.mak

TARGET_PREFIXES := f fh t th nb gh hclang
ALL_TARGETS := $(BASE_TARGET) $(patsubst %,%_$(BASE_TARGET),$(TARGET_PREFIXES)) $(EXTRA_TARGETS)

COMMON_FLAGS := -O3 -I../common

CC  = gcc
CXX = g++

.PHONY: all
all: $(ALL_TARGETS)

CC_CMD  = $(CC) $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(PROJECT_LDLIBS)
CXX_CMD = $(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(PROJECT_LDLIBS)

%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/default
%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

f_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/fibers
f_%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

fh_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/fibers-help
fh_%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

t_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/threads
t_%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

th_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/threads-help
th_%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

gh_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
gh_%: gh_%.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)
# fall back to default implementation if gh_*.cpp doesn't exit
gh_%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

nb_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
nb_%: nb_%.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)
# fall back to default implementation if nb_*.cpp doesn't exit
nb_%: %.cpp $(PROJECT_EXTRA_DEPS)
	$(CXX_CMD)

hclang_%: %.hc $(PROJECT_EXTRA_DEPS)
	rhcc $(COMMON_FLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf *.o $(ALL_TARGETS) *.dSYM *.out rose_*.c
