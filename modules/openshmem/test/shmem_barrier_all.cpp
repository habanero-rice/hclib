#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        hclib::locale_t *pe = hclib::shmem_my_pe();
        std::cout << "Hello world from rank " << hclib::pe_for_locale(pe) << std::endl;

        std::cout << "Rank " << hclib::pe_for_locale(pe) << " before barrier" << std::endl;
        hclib::shmem_barrier_all();
        std::cout << "Rank " << hclib::pe_for_locale(pe) << " after barrier" << std::endl;
    });
    return 0;
}
