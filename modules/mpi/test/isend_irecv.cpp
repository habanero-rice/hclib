#include "hclib_cpp.h"
#include "hclib_mpi.h"

#include <assert.h>
#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] () {
        int mpi_rank;
        hclib::MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        int nranks;
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);

        std::cout << "Hello world from rank " << mpi_rank << " out of " <<
            nranks << " ranks" << std::endl;

        int data = mpi_rank;

        assert(nranks % 2 == 0);

        if (mpi_rank % 2 == 0) {
            std::cout << "Rank " << mpi_rank << " sending async msg to " <<
                (mpi_rank + 1) << std::endl;
            hclib::future_t<void> *send_fut = hclib::MPI_Isend(&data, 1, MPI_INT,
                mpi_rank + 1, 0, MPI_COMM_WORLD);
            send_fut->wait();
        } else {
            std::cout << "Rank " << mpi_rank << " receiving async msg from " <<
                (mpi_rank - 1) << std::endl;
            hclib::future_t<void> *recv_fut = hclib::MPI_Irecv(&data, 1, MPI_INT,
                mpi_rank - 1, 0, MPI_COMM_WORLD);
            recv_fut->wait();
            assert(data == mpi_rank - 1);
        }
    });
    return 0;
}
