#include "hclib_cpp.h"
#include "hclib_mpi.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        hclib::locale_t *rank = hclib::MPI_Comm_rank(MPI_COMM_WORLD);
        std::cout << "Hello world from rank " << hclib::integer_rank_for_locale(rank) << std::endl;
    });
    return 0;
}
