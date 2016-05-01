#   ---------------------------
#   Compiler Flags
#   ---------------------------

#MPICC = mpixlc_r 
#MPICXX = mpicxx
MPICC = mpicc 
MPICXX = mpicxx

#LIB_TIMEBASE_PATH = /home/morozov/lib
MPI_CFLAGS = -I. -O3 -g -pg -DKMI_CONFIG_BINARY_STRING -DKMI_CONFIG_SORTING -DKMI_CONFIG_INDEXED_ARRAY -DKMI_CONFIG_SAMPLING_ITERATION -DKMI_CONFIG_INDEXED_ARRAY_HASH -DKMI_PROFILING_ALLTOALL 
Q_AR = ar

all: libkmi.a

KMI_C_OBJ = binary.o  db.o  debug.o  hash_table.o  init.o  iquery.o  malloc.o  partition.o  query.o  sorted_array.o  string.o  table.o 

$(KMI_C_OBJ): %.o: %.c
	$(MPICC) $(MPI_CFLAGS)-c $< -o $@

libkmi.a: $(KMI_C_OBJ)
	$(Q_AR) rcs $@ $(KMI_C_OBJ)

clean:
	rm -f *.a *.o a.out core*
