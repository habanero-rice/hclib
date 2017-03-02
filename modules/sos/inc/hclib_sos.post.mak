# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
HCLIB_SOS_CFLAGS+=-I$(HCLIB_ROOT)/../modules/sos/inc -I$(SOS_INSTALL)/include
HCLIB_SOS_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/sos/inc -I$(SOS_INSTALL)/include
HCLIB_SOS_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/sos/lib -L$(SOS_INSTALL)/lib
HCLIB_SOS_LDLIBS+=-lhclib_system -lhclib_sos
