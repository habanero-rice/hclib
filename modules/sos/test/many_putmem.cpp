#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <limits.h>

#include <iostream>

#define NPUTS 3000

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };
    hclib::launch(deps, 2, [argv] {
        const unsigned long long start_time = hclib_current_time_ns();

        const int npes = hclib::shmem_n_pes();
        const int pe = hclib::shmem_my_pe();
        int *buf = (int *)hclib::shmem_malloc(npes * sizeof(int));
        assert(buf);
        buf[pe] = pe;

        for (int i = 0; i < NPUTS; i++) {
            for (int j = 0; j < npes; j++) {
                hclib::shmem_putmem(buf + pe, buf + pe, sizeof(int), j);
            }
        }

        const unsigned long long elapsed = hclib_current_time_ns() - start_time;
        fprintf(stderr, "%s body took %f ms on PE %d\n", argv[0],
            (double)elapsed / 1000000.0, hclib::shmem_my_pe());
    });
    return 0;
}
