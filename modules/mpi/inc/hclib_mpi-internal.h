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

locale_t *MPI_Comm_rank(MPI_Comm comm);
void MPI_Comm_size(MPI_Comm comm, int *size);
locale_t *MPI_Comm_remote(MPI_Comm comm, int remote_rank);
int integer_rank_for_locale(locale_t *locale);

void MPI_Send(const void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *dest, int tag, MPI_Comm comm);
void MPI_Recv(void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *source, int tag, MPI_Comm comm, MPI_Status *status);

hclib::future_t *MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *dest, int tag, MPI_Comm comm);
hclib::future_t *MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *source, int tag, MPI_Comm comm);

}

#endif
