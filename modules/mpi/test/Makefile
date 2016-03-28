include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/../modules/mpi/inc/hclib_mpi.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include $(HCLIB_ROOT)/../modules/mpi/inc/hclib_mpi.post.mak

TARGETS=init send_recv isend_irecv

all: $(TARGETS)

%: %.cpp
	g++ -std=c++11 $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS) -lmpi_cxx

clean:
	rm -f $(TARGETS)
