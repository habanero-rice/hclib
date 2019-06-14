include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak

TARGETS=async0 async1 finish0 finish1 finish2  forasync1DCh  forasync1DRec \
		forasync2DCh  forasync2DRec  forasync3DCh  forasync3DRec \
		promise/asyncAwait0 promise/asyncAwait0Null promise/future0 \
		promise/future1 promise/future2 promise/future3 promise/future4 neconlce1 access_argc \
		promise/asyncAwait0Shared promise/asyncAwait0Unique \
		promise/future0Float promise/future0Int \
		no_async_finish nested_finish nested_finish_async_await future_wait_in_finish atomic atomic_sum \
		capture0 capture1 copies0 copies1 promise/async_future_await_at promise/asyncAwait0Vector

FLAGS=-g -std=c++11 -Wall

all: $(TARGETS) targets.txt

.PHONY: targets.txt # always update
targets.txt:
	@echo "$(TARGETS)" > $@

%: %.cpp
	$(CXX) ${FLAGS} $(HCLIB_CFLAGS) $(HCLIB_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS)

clean:
	rm -f $(TARGETS)
