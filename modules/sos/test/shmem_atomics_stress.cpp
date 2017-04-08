#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <iostream>

// long lock = 0L;
static int pe = -1;

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };

    hclib::launch(deps, 2, [] {
        pe = hclib::shmem_my_pe();

        long *target = (long *)shmem_malloc(sizeof(long));
        assert(target);
        *target = 0;
        hclib::shmem_barrier_all();

        const int expect = pe % 2;
        int set_to;
        if (expect == 0) {
            set_to = 1;
        } else {
            set_to = 0;
        }

        hclib::finish([&] {
            for (int i = 0; i < 10; i++) {
                hclib::async([=] {
                    const int ret = hclib::shmem_long_finc(target, 0);
                });
            }
        });

        fprintf(stderr, "%d out of finish\n", ::shmem_my_pe());

        hclib::shmem_barrier_all();
        hclib::shmem_free(target);
    });

    return 0;
}
