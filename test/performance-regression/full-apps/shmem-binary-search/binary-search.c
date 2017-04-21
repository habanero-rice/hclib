/*
 *  Copyright (c) 2016 Intel Corporation. All rights reserved.
 *  This software is available to you under the BSD license below:
 *
 *      Redistribution and use in source and binary forms, with or
 *      without modification, are permitted provided that the following
 *      conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <shmemx.h>
#include <assert.h>

int ipWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double dpWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[SHMEM_REDUCE_SYNC_SIZE];

static int binary_search(const int key, int *keys, const int keys_per_pe) {
    int low, mid, high;

    low = 0;
    high = shmem_n_pes() * keys_per_pe;

    while(low < high) {
        int val;

        mid = low + (high-low)/2;
        val = shmem_int_g(&keys[mid % keys_per_pe], mid / keys_per_pe);

        if(val == key) {
            return mid;
        } else if(val < key) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return -1;
}

int main(int argc, char **argv) {
    int i;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <keys-per-pe>\n", argv[0]);
        return 1;
    }

    const int keys_per_pe = atoi(argv[1]);

    shmem_init();

    const int pe = shmem_my_pe();
    const int npes = shmem_n_pes();

    for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        ipWrk[i] = SHMEM_SYNC_VALUE;
        dpWrk[i] = SHMEM_SYNC_VALUE;
    }

    int *keys = (int *)shmem_malloc(keys_per_pe * sizeof(int));
    assert(keys);
    int *my_errs = (int *)shmem_malloc(sizeof(int));
    assert(my_errs);
    int *all_errs = (int *)shmem_malloc(sizeof(int));
    assert(all_errs);

    double *my_elapsed_time = (double *)shmem_malloc(sizeof(double));
    double *cumulative_elapsed_time = (double *)shmem_malloc(sizeof(double));
    assert(my_elapsed_time && cumulative_elapsed_time);

    *my_errs = 0;

    for (i = 0; i < keys_per_pe; i++)
        keys[i] = keys_per_pe * pe + i;

    shmem_barrier_all();

    const double start_time = shmemx_wtime();

    for (i = 0; i < keys_per_pe * shmem_n_pes(); i++) {
        int j = binary_search(i, keys, keys_per_pe);
        if (j != i) {
            printf("%2d: Error searching for %d.  Found at index %d, expected "
                    "%d\n", pe, i, j, i);
            *my_errs += 1;
        }
    }

    *my_elapsed_time = shmemx_wtime() - start_time;

    shmem_int_sum_to_all(all_errs, my_errs, 1, 0, 0, npes, ipWrk, pSync);
    shmem_barrier_all();
    shmem_double_sum_to_all(cumulative_elapsed_time, my_elapsed_time, 1, 0, 0,
            npes, dpWrk, pSync);

    if (pe == 0) {
        if (*all_errs == 0) {
            printf("Validated with %d keys per PE, %d keys in total!\n",
                    keys_per_pe, keys_per_pe * npes);
            printf("Cumulative time : %f s\n", *cumulative_elapsed_time);
            printf("Wall time : %f s\n", *my_elapsed_time);
        }
    }

    shmem_finalize();

    return 0;
}
