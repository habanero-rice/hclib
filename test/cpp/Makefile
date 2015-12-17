include $(HCPP_ROOT)/include/hcpp.mak

TARGETS=async0 async1 finish0 finish1 finish2  forasync1DCh  forasync1DRec \
		forasync2DCh  forasync2DRec  forasync3DCh  forasync3DRec \
		ddf/asyncAwait0 ddf/asyncAwait0Null ddf/asyncAwait1 ddf/future0 \
		ddf/future1 ddf/future2 ddf/future3

FLAGS=-g -std=c++11

all: $(TARGETS)

%: %.cpp
	g++ ${FLAGS} $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
