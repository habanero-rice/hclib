# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.

HCLIB_OSHMEM_CFLAGS+=-I$(HCLIB_ROOT)/../modules/openshmem-am/inc -I$(OPENSHMEM_INSTALL)/include
HCLIB_OSHMEM_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/openshmem-am/inc -I$(OPENSHMEM_INSTALL)/include
HCLIB_OSHMEM_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/openshmem-am/lib -L$(OPENSHMEM_INSTALL)/lib
HCLIB_OSHMEM_LDLIBS+=-lrt -lhclib_system -lhclib_openshmem-am -loshmem $(GASNET_LIBS)
