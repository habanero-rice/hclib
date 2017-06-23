# default to release build
PROJECT_GOALS := release

include $(HCLIB_ROOT)/include/hclib.mak

ALL_TARGETS := $(BASE_TARGET) nb_$(BASE_TARGET) hclang_$(BASE_TARGET)

COMMON_FLAGS := -O3

.PHONY: all
all: $(ALL_TARGETS)

CXX_CMD = $(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(COMMON_FLAGS) -o $@ $^ $(PROJECT_LDLIBS)

%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/default
%: %.cpp
	$(CXX_CMD)

nb_%: HCLIB_ROOT:=$(HCLIB_ROOT)/multi/non-blocking
nb_%: nb_%.cpp
	$(CXX_CMD)

hclang_%: %.hc
	hcc $(COMMON_FLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -rf *.o $(ALL_TARGETS) *.dSYM *.out rose_*.c
