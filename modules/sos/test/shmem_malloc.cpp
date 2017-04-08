#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };
    hclib::launch(deps, 2, [] {
        std::cout << "Hello world from rank " << hclib::shmem_my_pe() << std::endl;

        int *allocated = (int *)hclib::shmem_malloc(10 * sizeof(int));
        assert(allocated);

        std::cout << "Rank " << hclib::shmem_my_pe() << " allocated " << allocated << std::endl;

        for (int i = 0; i < 10; i++) {
            allocated[i] = i;
        }

        hclib::shmem_free(allocated);
    });
    return 0;
}
