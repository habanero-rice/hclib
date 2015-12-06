include $(HCPP_ROOT)/include/hcpp.mak

TARGETS=async0 async1 finish0 finish1 finish2  forasync1DCh  forasync1DRec  forasync2DCh  forasync2DRec  forasync3DCh  forasync3DRec

all: $(TARGETS)

%: %.c
	gcc $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -f $(TARGETS)
