include $(HCLIB_ROOT)/include/hclib.mak

TARGETS=async0 async1 finish0 finish1 finish2  forasync1DCh  forasync1DRec \
		forasync2DCh  forasync2DRec  forasync3DCh  forasync3DRec \
		promise/asyncAwait0 promise/asyncAwait0Null promise/asyncAwait1 promise/future0 \
		promise/future1 promise/future2 promise/future3 neconlce1 access_argc \
		promise/asyncAwait0Shared promise/asyncAwait0Unique \
		promise/future0Float promise/future0Int \
		capture0 capture1 copies0 copies1

FLAGS=-g -std=c++11

all: $(TARGETS) targets.txt

.PHONY: targets.txt # always update
targets.txt:
	@echo "$(TARGETS)" > $@

%: %.cpp
	g++ ${FLAGS} $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
