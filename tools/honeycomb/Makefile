include $(HCLIB_ROOT)/include/hclib.mak

CXX=honey64-unknown-hcos-g++

TARGETS=async0

FLAGS=-g -std=c++11 -Wall

all: $(TARGETS) targets.txt

.PHONY: targets.txt # always update
targets.txt:
	@echo "$(TARGETS)" > $@

HCLIB_SYSTEM=-I $(HCLIB_ROOT)/../modules/system/inc $(HCLIB_ROOT)/../modules/system/src/hclib_system.cpp

%: %.cpp
	$(CXX) ${FLAGS} $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_SYSTEM) -o $@ $^ -lhclib -lpthread

clean:
	rm -f $(TARGETS)

test:
	sim-safe -quiet ./async0
