#ifndef HCLIB_MPI_INTERNAL_H
#define HCLIB_MPI_INTERNAL_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#include <mpi.h>

#include <stdlib.h>
#include <stdio.h>

#define CHECK_MPI(call) { \
    const int err = (call); \
    if (err != MPI_SUCCESS) { \
        fprintf(stderr, "MPI ERR %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
}

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(mpi_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(mpi_post_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(mpi_finalize);

void MPI_Comm_rank(MPI_Comm comm, int *rank);
void MPI_Comm_size(MPI_Comm comm, int *size);

void MPI_Send(void *buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm);
void MPI_Recv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Status *status);

hclib::future_t<void> *MPI_Isend(void *buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm);
hclib::future_t<void> *MPI_Isend_await(void *buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm, hclib::future_t<void> *fut);
hclib::future_t<void> *MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm);
hclib::future_t<void> *MPI_Irecv_await(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, hclib::future_t<void> *fut);

void MPI_Waitall(int count, hclib::future_t<void> *array_of_requests[]);

inline double MPI_Wtime(void) {
    return ::MPI_Wtime();
}

void MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
void MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);

void MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
hclib::future_t<void> *MPI_Allreduce_future(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
void MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, 
        MPI_Comm comm);
void MPI_Barrier(MPI_Comm comm);

void MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
}
#endif
