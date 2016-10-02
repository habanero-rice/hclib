include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/upcxx/inc/hclib_upcxx.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/upcxx/inc/hclib_upcxx.post.mak

TARGETS=basic active_msg

CC?=/opt/gcc/4.9.0/bin/gcc

all: $(TARGETS)

%: %.cpp
	$(CC) -std=c++11 $(HCLIB_CFLAGS) $(HCLIB_UPCXX_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_UPCXX_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_UPCXX_LDLIBS)

%: %.c
	$(CC) $(HCLIB_CFLAGS) $(HCLIB_UPCXX_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_UPCXX_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_UPCXX_LDLIBS)


clean:
	rm -f $(TARGETS)
