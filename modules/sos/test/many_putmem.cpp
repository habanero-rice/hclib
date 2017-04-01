#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <limits.h>

#include <iostream>

#define NPUTS 600

void sig_handler(int signo) {
    raise(SIGABRT);
    assert(0); // should never reach here
}

void *kill_func(void *data) {
    int kill_seconds = *((int *)data);
    int err = sleep(kill_seconds);
    assert(err == 0);
    raise(SIGUSR1);
    return NULL;
}

int main(int argc, char **argv) {
    // __sighandler_t serr = signal(SIGUSR1, sig_handler);
    // assert(serr != SIG_ERR);
    // pthread_t thread;
    // int kill_seconds = 180;
    // const int perr = pthread_create(&thread, NULL, kill_func,
    //         (void *)&kill_seconds);
    // assert(perr == 0);

    const char *deps[] = { "system", "sos" };
    hclib::launch(deps, 2, [argv] {
        const unsigned long long start_time = hclib_current_time_ns();

        const int npes = hclib::shmem_n_pes();
        const int pe = hclib::shmem_my_pe();
        int *buf = (int *)hclib::shmem_malloc(npes * sizeof(int));
        assert(buf);
        buf[pe] = pe;

        // hclib::finish([&] {
            for (int i = 0; i < NPUTS; i++) {
                for (int j = 0; j < npes; j++) {
                    // hclib::shmem_putmem(buf + pe, buf + pe, sizeof(int), j);
                    // hclib::shmem_putmem(buf + pe, buf + pe, sizeof(int), j);
                    hclib::shmem_putmem(buf, buf, sizeof(int), j);
                }
            }
        // });

        const unsigned long long elapsed = hclib_current_time_ns() - start_time;
        fprintf(stderr, "%s body took %f ms\n", argv[0],
            (double)elapsed / 1000000.0);
    });
    return 0;
}
