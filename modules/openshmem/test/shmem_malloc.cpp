#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        hclib::locale_t *pe = hclib::shmem_my_pe();
        std::cout << "Hello world from rank " << hclib::pe_for_locale(pe) << std::endl;

        int *allocated = (int *)hclib::shmem_malloc(10 * sizeof(int));
        std::cout << "Rank " << hclib::pe_for_locale(pe) << " allocated " << allocated << std::endl;

        for (int i = 0; i < 10; i++) {
            allocated[i] = i;
        }
    });
    return 0;
}
