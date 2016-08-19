include $(HCLIB_ROOT)/include/hclib.mak

TARGETS=allocate copy kernel

FLAGS=-g -std=c++11

all: $(TARGETS)

kernel: kernel.cu
	nvcc -arch=sm_30 ${FLAGS} $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

%: %.cpp
	$(CXX) ${FLAGS} $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
