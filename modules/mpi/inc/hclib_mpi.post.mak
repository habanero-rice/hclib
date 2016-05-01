# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
HCLIB_MPI_CFLAGS+=-I$(HCLIB_ROOT)/../modules/mpi/inc -I$(MPI_HOME)/include
HCLIB_MPI_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/mpi/inc -I$(MPI_HOME)/include
HCLIB_MPI_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/mpi/lib -L$(MPI_HOME)/lib
HCLIB_MPI_LDLIBS+=-lhclib_mpi -lmpi
