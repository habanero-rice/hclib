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
            std::cout << "Rank " << mpi_rank << " sending message to " <<
                (mpi_rank + 1) << std::endl;
            hclib::MPI_Send(&data, 1, MPI_INT,
                mpi_rank + 1, 0, MPI_COMM_WORLD);
        } else {
            std::cout << "Rank " << mpi_rank << " receiving message from " <<
                (mpi_rank - 1) << std::endl;
            MPI_Status status;
            hclib::MPI_Recv(&data, 1, MPI_INT,
                mpi_rank - 1, 0, MPI_COMM_WORLD, &status);
            assert(data == mpi_rank - 1);
        }
    });
    return 0;
}
