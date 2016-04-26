include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=async0 async1 finish0 finish1 finish2  forasync1DCh  forasync1DRec \
		forasync2DCh  forasync2DRec  forasync3DCh  forasync3DRec \
		promise/asyncAwait0 promise/asyncAwait0Null promise/asyncAwait1 promise/future0 \
		promise/future1 promise/future2 promise/future3 neconlce1 access_argc \
		no_async_finish nested_finish future_wait_in_finish

FLAGS=-g -std=c++11

all: $(TARGETS)

%: %.cpp
	g++ ${FLAGS} $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean:
	rm -f $(TARGETS)
