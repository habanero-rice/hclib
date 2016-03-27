# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
PROJECT_CFLAGS+=-I$(HCLIB_ROOT)/../modules/mpi/inc
PROJECT_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/mpi/inc
PROJECT_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/mpi/lib -L$(MPI_HOME)/lib
PROJECT_LDLIBS += $(call GET_LINK_FLAG,--whole-archive) \
				  $(call GET_LINK_FLAG,-lhclib_mpi) \
				  $(call GET_LINK_FLAG,--no-whole-archive) \
				  $(call GET_LINK_FLAG,-lmpi)
