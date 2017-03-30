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

        const unsigned long long start_time = hclib_current_time_ns();
        hclib::finish([&] {
            for (int i = 0; i < NREPEATS; i++) {
                // hclib::async([] {
                    hclib::shmem_barrier_all();
                // });
            }
        });
        const unsigned long long elapsed = hclib_current_time_ns() - start_time;
        fprintf(stderr, "PE %d elapsed time = %f ms\n", shmem_my_pe(),
            (double)elapsed / 1000000.0);
    });

    return 0;
}
