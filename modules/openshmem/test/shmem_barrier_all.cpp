#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system", "openshmem" };
    hclib::launch(deps, 2, [] {
        std::cout << "Hello world from rank " << hclib::shmem_my_pe() << std::endl;

        std::cout << "Rank " << hclib::shmem_my_pe() << " before barrier" << std::endl;
        hclib::shmem_barrier_all();
        std::cout << "Rank " << hclib::shmem_my_pe() << " after barrier" << std::endl;
    });
    return 0;
}
