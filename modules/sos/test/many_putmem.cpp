#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <iostream>

#define NPUTS 10000

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };
    hclib::launch(deps, 2, [] {

        const int npes = hclib::shmem_n_pes();
        const int pe = hclib::shmem_my_pe();
        int *buf = (int *)hclib::shmem_malloc(npes * sizeof(int));
        assert(buf);
        buf[pe] = pe;
       
        for (int i = 0; i < NPUTS; i++) {
            for (int j = 0; j < npes; j++) {
                hclib::async([=] {
                    hclib::shmem_putmem(buf + pe, buf + pe, sizeof(int), j);
                });
            }
        }
    });
    return 0;
}
