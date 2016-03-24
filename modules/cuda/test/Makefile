include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/cuda/inc/hclib_cuda.mak

TARGETS=init

all: $(TARGETS)

%: %.cpp
	g++ -std=c++11 $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
