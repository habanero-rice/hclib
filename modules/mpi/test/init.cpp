#include "hclib_cpp.h"
#include "hclib_mpi.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] () {
        int rank;
        hclib::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "Hello world from rank " << rank << std::endl;
    });
    return 0;
}
