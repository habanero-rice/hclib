#include "hclib_cpp.h"
#include "hclib_mpi.h"
#include "hclib_system.h"

#include <assert.h>
#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        hclib::locale_t *rank = hclib::MPI_Comm_rank(MPI_COMM_WORLD);
        const int mpi_rank = hclib::integer_rank_for_locale(rank);

        int nranks;
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);

        std::cout << "Hello world from rank " << mpi_rank << " out of " <<
            nranks << " ranks" << std::endl;

        int data = mpi_rank;

        assert(nranks % 2 == 0);

        if (hclib::integer_rank_for_locale(rank) % 2 == 0) {
            std::cout << "Rank " << mpi_rank << " sending async msg to " <<
                (mpi_rank + 1) << std::endl;
            hclib::future_t *send_fut = hclib::MPI_Isend(&data, 1, MPI_INT,
                hclib::MPI_Comm_remote(MPI_COMM_WORLD, mpi_rank + 1), 0,
                MPI_COMM_WORLD);
            send_fut->wait();
        } else {
            std::cout << "Rank " << mpi_rank << " receiving async msg from " <<
                (mpi_rank - 1) << std::endl;
            hclib::future_t *recv_fut = hclib::MPI_Irecv(&data, 1, MPI_INT,
                hclib::MPI_Comm_remote(MPI_COMM_WORLD, mpi_rank - 1), 0,
                MPI_COMM_WORLD);
            recv_fut->wait();
            assert(data == mpi_rank - 1);
        }
    });
    return 0;
}
