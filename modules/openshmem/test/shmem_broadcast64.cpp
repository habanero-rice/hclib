#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system", "openshmem" };
    hclib::launch(deps, 2, [] {
        const int pe = hclib::shmem_my_pe();
        std::cout << "Hello world from rank " << pe << std::endl;

        uint64_t *shared_source = (uint64_t *)hclib::shmem_malloc(sizeof(uint64_t));
        uint64_t *shared_dest = (uint64_t *)hclib::shmem_malloc(sizeof(uint64_t));
        long *pSync = (long *)hclib::shmem_malloc(sizeof(long) * SHMEM_BCAST_SYNC_SIZE);

        const size_t n_pes = hclib::shmem_n_pes();
        for (int root = 0; root < n_pes; root++) {
            *shared_source = pe;
            hclib::shmem_barrier_all();

            for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
                pSync[i] = SHMEM_SYNC_VALUE;
            }

            hclib::shmem_broadcast64(shared_dest, shared_source, 1, root, 0, 0, n_pes, pSync);

            if (pe != root) {
                assert(*shared_dest == root);
            }
        }
        fprintf(stderr, "PE %d done\n", pe);

    });
    return 0;
}
