#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <signal.h>
#include <unistd.h>

#include <iostream>

// long lock = 0L;
static int pe = -1;

unsigned long long get_clock_gettime() {
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (unsigned long long)t.tv_nsec + s;
}

void *kill_func(void *data) {
    int kill_seconds = *((int *)data);
    int err = sleep(kill_seconds);
    assert(err == 0);
    raise(SIGABRT);
    return NULL;
}

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };
    int kill_seconds = 120;
    pthread_t thread;
    const int perr = pthread_create(&thread, NULL, kill_func,
            (void *)&kill_seconds);
    assert(perr == 0);

    hclib::launch(deps, 2, [] {
        pe = hclib::shmem_my_pe();

        long *lock = (long *)shmem_malloc(sizeof(long));
        *lock = 0;
        hclib::shmem_barrier_all();

        const unsigned long long start_time = get_clock_gettime();
        hclib::finish([=] {

            size_t nlocks = 0;
            for (nlocks = 0; nlocks < 50000; nlocks++) {
                hclib::async([=] {
                    hclib::shmem_set_lock(lock);
                    hclib::shmem_clear_lock(lock);
                });
            }

            const unsigned long long issued_ns = get_clock_gettime() -
                    start_time;
            fprintf(stderr, "PE %d issued %lu locks in %d ms\n", pe, nlocks,
                    issued_ns / 1000000ULL);
        });
        const unsigned long long elapsed_ns = get_clock_gettime() -
                start_time;
        fprintf(stderr, "PE %d completed in %d ms\n", pe,
                elapsed_ns / 1000000ULL);

        fprintf(stderr, "PE %d out of finish\n", pe);

        hclib::shmem_barrier_all();
        hclib::shmem_free(lock);
    });

    return 0;
}
