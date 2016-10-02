#include <stdio.h>

#include "hclib_cpp.h"
#include "hclib_upcxx.h"
#include "hclib_system.h"

#ifndef N
#define N 4096
#endif

int arr[N];

void handler(int i) {
    arr[i] = i;
}

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    int err = 1;

    hclib::launch(deps, 1, [&] {
        int myrank = hclib::upcxx::myrank();
        int ranks = hclib::upcxx::ranks();
        int neighbor = (myrank + 1) % ranks;
        assert(ranks > 1);
        memset(arr, 0x00, N * sizeof(int));

        hclib::upcxx::barrier();

        hclib::upcxx::remote_finish([&] {
            for (int i = 0; i < N; i++) {
                hclib::upcxx::async(neighbor, [=] {
                    handler(i);
                });
            }
        });

        while (err == 1) {
            err = 0;

            hclib::upcxx::advance();

            for (int i = 0; i < N; i++) {
                if (arr[i] != i) {
                    err = 1;
                    break;
                }
            }
        }

        fprintf(stderr, "Rank %d complete\n", hclib::upcxx::myrank());
    });

    return err;
}
