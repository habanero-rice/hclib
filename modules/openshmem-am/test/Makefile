include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/openshmem-am/inc/hclib_openshmem-am.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/openshmem-am/inc/hclib_openshmem-am.post.mak

TARGETS=init

all: $(TARGETS)

%: %.cpp
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c++11 $^ -I$(HCLIB_ROOT)/include \
		-I$(HCLIB_ROOT)/../modules/openshmem-am/inc -I$(HCLIB_ROOT)/../modules/openshmem/inc -I$(HCLIB_ROOT)/../modules/system/inc \
		-lstdc++ -lhclib -lhclib_openshmem -lhclib_openshmem-am -L$(HCLIB_ROOT)/lib -L$(HCLIB_HOME)/modules/openshmem/lib -L../lib $(JSMN_HOME)/libjsmn.a -o $@
	# g++ -std=c++11 $(HCLIB_CFLAGS) $(HCLIB_OSHMEM_CFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_OSHMEM_LDFLAGS) -o $@ $^ $(HCLIB_LDLIBS) $(HCLIB_OSHMEM_LDLIBS)

clean:
	rm -f $(TARGETS)
