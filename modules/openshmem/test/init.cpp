#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        hclib::locale_t *pe = hclib::shmem_my_pe();
        std::cout << "Hello world from rank " << hclib::pe_for_locale(pe) << std::endl;
    });
    return 0;
}
