include $(HCPP_ROOT)/include/hcpp.mak

TARGETS=allocate copy kernel

FLAGS=-g -std=c++11

all: $(TARGETS)

kernel: kernel.cu
	nvcc -arch=sm_30 ${FLAGS} $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

%: %.cpp
	g++ ${FLAGS} $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
