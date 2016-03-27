#ifndef HCLIB_MPI_H
#define HCLIB_MPI_H

#include "hclib_mpi-internal.h"

namespace hclib {

template<typename... future_list_t>
hclib::future_t *MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm, future_list_t... futures) {
}

template<typename... future_list_t>
hclib::future_t *MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, future_list_t... futures) {
}

}

#endif
