include $(HCLIB_ROOT)/include/hclib.mak
# Depends on the system module
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.mak
include $(HCLIB_ROOT)/../modules/cuda/inc/hclib_cuda.mak

TARGETS=init allocate

all: $(TARGETS)

%: %.cpp
	g++ -std=c++11 $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
