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

#if defined(HCLIB_MEASURE_START_LATENCY) || defined(HCLIB_PROFILE)
enum MPI_FUNC_LABELS {
    MPI_Send_lbl = 0,
    MPI_Recv_lbl,
    MPI_Isend_lbl,
    MPI_Irecv_lbl,
    MPI_Allreduce_lbl,
    MPI_Allreduce_future_lbl,
    MPI_Bcast_lbl,
    MPI_Barrier_lbl,
    MPI_Allgather_lbl,
    MPI_Reduce_lbl,
    MPI_Waitall_lbl,
    N_MPI_FUNCS
};
#endif

#ifdef HCLIB_PROFILE
#define MPI_START_PROFILE const unsigned long long __mpi_profile_start_time = hclib_current_time_ns();

#define MPI_END_PROFILE(funcname) { \
    const unsigned long long __mpi_profile_end_time = hclib_current_time_ns(); \
    mpi_profile_counters[funcname##_lbl]++; \
    mpi_profile_times[funcname##_lbl] += (__mpi_profile_end_time - __mpi_profile_start_time); \
}
#else
#define MPI_START_PROFILE
#define MPI_END_PROFILE(funcname)
#endif

#ifdef HCLIB_MEASURE_START_LATENCY
#define MPI_START_LATENCY const unsigned long long __mpi_latency_start_time = hclib_current_time_ns();
#define MPI_END_LATENCY(funcname) { \
    const unsigned long long __mpi_latency_end_time = hclib_current_time_ns(); \
    mpi_latency_counters[funcname##_lbl]++; \
    mpi_latency_times[funcname##_lbl] += (__mpi_latency_end_time - __mpi_latency_start_time); \
}
#else
#define MPI_START_LATENCY
#define MPI_END_LATENCY(funcname)
#endif

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

hclib::future_t *MPI_Isend(void *buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm);
hclib::future_t *MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm);

void MPI_Waitall(int count, hclib::future_t *array_of_requests[]);

inline double MPI_Wtime(void) {
    return ::MPI_Wtime();
}

void MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
void MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);

void MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
hclib::future_t *MPI_Allreduce_future(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
void MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, 
        MPI_Comm comm);
void MPI_Barrier(MPI_Comm comm);

void MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);


void print_mpi_profiling_data();

}

#endif
