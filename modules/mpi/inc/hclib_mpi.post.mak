# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
HCLIB_MPI_CFLAGS+=-I$(HCLIB_ROOT)/../modules/mpi/inc -I$(MPI_HOME)/include
HCLIB_MPI_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/mpi/inc -I$(MPI_HOME)/include
HCLIB_MPI_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/mpi/lib -L$(MPI_HOME)/lib

ifeq ($(IS_MAC_OS),1)
HCLIB_MPI_LDLIBS += -lmpi $(call GET_LINK_FLAG,-force_load) \
				  $(call GET_LINK_FLAG,$(HCLIB_ROOT)/../modules/mpi/lib/libhclib_mpi.a)
else
HCLIB_MPI_LDLIBS += $(call GET_LINK_FLAG,--whole-archive) \
				  $(call GET_LINK_FLAG,$(HCLIB_ROOT)/../modules/mpi/lib/libhclib_mpi.a) \
				  $(call GET_LINK_FLAG,--no-whole-archive) \
				  $(call GET_LINK_FLAG,-lmpi)
endif
