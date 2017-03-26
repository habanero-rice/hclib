CFLAGS = -L$(OPENSHMEM_INSTALL)/lib -lmpi -I$(OPENSHMEM_INSTALL)/include -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math -DGRAPH_GENERATOR_MPI -DGENERATOR_USE_PACKED_EDGE_TYPE # -g -pg
# CFLAGS = -g -Wall -Drestrict= -DGRAPH_GENERATOR_MPI
LDFLAGS = -fopenmp -g # -g -pg
MPICC = gcc

all: graph500_mpi_simple graph500_mpi_one_sided graph500_mpi_replicated graph500_mpi_replicated_csc graph500_mpi_custom graph500_mpi_replicated_hiper

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES = main.c oned_csr.c oned_csc.c utils.c validate.c onesided.c onesided_emul.c
HEADERS = common.h oned_csr.h oned_csc.h redistribute.h mpi_workarounds.h onesided.h

graph500_mpi_simple: bfs_simple.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_simple bfs_simple.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_one_sided: bfs_one_sided.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_one_sided bfs_one_sided.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated: bfs_replicated.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated bfs_replicated.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated_hiper: bfs_replicated_hiper.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	g++ -std=c++11 $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated_hiper bfs_replicated_hiper.c $(SOURCES) $(GENERATOR_SOURCES) -lm -I$(HCLIB_ROOT)/include -I$(HCLIB_HOME)/modules/system/inc -I$(HCLIB_HOME)/modules/mpi/inc -L$(HCLIB_ROOT)/lib -L$(HCLIB_HOME)/modules/mpi/lib -lhclib -lhclib_mpi $(JSMN_HOME)/jsmn.c

graph500_mpi_replicated_csc: bfs_replicated_csc.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated_csc bfs_replicated_csc.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_custom: bfs_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_custom bfs_custom.c $(SOURCES) $(GENERATOR_SOURCES) -lm

clean:
	-rm -f graph500_mpi_* *.o *.a
