#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <iostream>

#define NREPEATS 1000

// long lock = 0L;
static int pe = -1;

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };

    hclib::launch(deps, 2, [] {
        pe = hclib::shmem_my_pe();


        hclib::finish([&] {
            for (int i = 0; i < NREPEATS; i++) {
                // hclib::async([] {
                    hclib::shmem_barrier_all();
                // });
            }
        });

        fprintf(stderr, "%d out of finish\n", ::shmem_my_pe());
    });

    return 0;
}
