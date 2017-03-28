#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <iostream>

// long lock = 0L;
static int pe = -1;

unsigned long long get_clock_gettime() {
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (unsigned long long)t.tv_nsec + s;
}

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };

    hclib::launch(deps, 2, [] {
        pe = hclib::shmem_my_pe();

        long *lock = (long *)shmem_malloc(sizeof(long));
        *lock = 0;
        hclib::shmem_barrier_all();

        hclib::finish([=] {
            const unsigned long long start_time = get_clock_gettime();

            size_t nlocks = 0;
            for (nlocks = 0; nlocks < 20000; nlocks++) {
                hclib::async([=] {
                    hclib::shmem_set_lock(lock);
                    hclib::shmem_clear_lock(lock);
                });
            }

            const unsigned long long elapsed_ns = get_clock_gettime() -
                    start_time;
            fprintf(stderr, "PE %d issued %lu locks in %d ms\n", pe, nlocks,
                    elapsed_ns / 1000000ULL);
        });

        fprintf(stderr, "PE %d out of finish\n", pe);

        hclib::shmem_barrier_all();
        hclib::shmem_free(lock);
    });

    return 0;
}
