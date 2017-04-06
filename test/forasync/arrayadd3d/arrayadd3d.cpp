/*
 * HC CONCORD foreach add2d.hc example
 */

#include "hclib.hpp"

void check(int *a, int val, int num_iters) {
    int i;
    for (i = 0; i < num_iters; i++) {
        if (a[i] != val) {
            printf("ERROR a[%d]=%d!=%d\n", i, a[i], val);
            exit(0);
        }
    }
}

int main(int argc, char *argv[]) {
    hclib::launch([&]() {
        int num_iters1;
        int num_iters2;
        int num_iters3;
        int tilesize1;
        int tilesize2;
        int tilesize3;
        int i;
        int *a, *b, *c;

        if (argc != 7) {
            printf("USAGE:./arrayadd2d NUM_ITERS1 NUM_ITERS2 TILE_SIZE1 "
                   "TILE_SIZE2\n");
            exit(1);
        }
        num_iters1 = atoi(argv[1]);
        num_iters2 = atoi(argv[2]);
        num_iters3 = atoi(argv[3]);
        tilesize1 = atoi(argv[4]);
        tilesize2 = atoi(argv[5]);
        tilesize3 = atoi(argv[6]);

        a = (int *)malloc(sizeof(int) * num_iters1 * num_iters2 * num_iters3);
        b = (int *)malloc(sizeof(int) * num_iters1 * num_iters2 * num_iters3);
        c = (int *)malloc(sizeof(int) * num_iters1 * num_iters2 * num_iters3);

        // Initialize the Values
        for (i = 0; i < num_iters1 * num_iters2 * num_iters3; i++) {
            a[i] = 0;
            b[i] = 100;
            c[i] = 1;
        }
        // Add the elements of arrays b and c and store them in a
        hclib::finish([=]() {
            hclib::loop_domain_t loop[3] = {{0, num_iters1, 1, tilesize1},
                                            {0, num_iters2, 1, tilesize2},
                                            {0, num_iters3, 1, tilesize3}};
            hclib::forasync3D(
                    loop,
                    [=](int i, int j, int k) {
                        a[i * num_iters2 * num_iters3 + j * num_iters3 + k] =
                                b[i * num_iters2 * num_iters3 + j * num_iters3 +
                                  k] +
                                c[i * num_iters2 * num_iters3 + j * num_iters3 +
                                  k];
                    },
                    FORASYNC_MODE_RECURSIVE);
            //}, FORASYNC_MODE_FLAT);
        });

        check(a, 101, num_iters1 * num_iters2 * num_iters3);
        printf("Test passed\n");
    });
    return 0;
}
