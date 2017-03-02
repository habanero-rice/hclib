include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/sos/inc/hclib_sos.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/sos/inc/hclib_sos.post.mak

TARGETS=init

all: $(TARGETS)

%: %.cpp
	g++ -std=c++11 $(HCLIB_CFLAGS) $(HCLIB_SOS_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_SOS_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_SOS_LDLIBS)

%: %.c
	gcc $(HCLIB_CFLAGS) $(HCLIB_SOS_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_SOS_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_SOS_LDLIBS)


clean:
	rm -f $(TARGETS)
