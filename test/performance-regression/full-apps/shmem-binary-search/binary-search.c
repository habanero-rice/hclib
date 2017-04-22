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

#ifdef USE_PTHREADS
#include <pthread.h>

#ifndef USE_THREAD_INITIALIZER
#error Pthreads version requires -DUSE_THREAD_INITIALIZER
#endif
#endif

#ifdef USE_CONTEXTS
#ifndef USE_THREAD_INITIALIZER
#error Contexts version requires -DUSE_THREAD_INITIALIZER
#endif
#ifndef USE_PTHREADS
#error Contexts version requires -DUSE_PTHREADS
#endif
#endif

int ipWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double dpWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[SHMEM_REDUCE_SYNC_SIZE];

typedef struct {
    int start_key;
    int end_key;
    int *keys;
    int keys_per_pe;
    int pe;
    int my_errs;
#ifdef USE_CONTEXTS
    shmemx_ctx_t ctx;
#endif
} thread_context;

static int binary_search(const int key, const int *keys, const int keys_per_pe
#ifdef USE_CONTEXTS
        , shmemx_ctx_t ctx
#endif
        ) {
    int mid;

    int low = 0;
    int high = shmem_n_pes() * keys_per_pe;

    while(low < high) {
        int val;

        mid = low + (high-low)/2;
#ifdef USE_CONTEXTS
        val = shmemx_ctx_int_g(&keys[mid % keys_per_pe], mid / keys_per_pe,
                ctx);
#else
        val = shmem_int_g(&keys[mid % keys_per_pe], mid / keys_per_pe);
#endif

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

static void *thread_body(void *input) {
    thread_context *data = (thread_context *)input;
    const int *keys = data->keys;
    const int keys_per_pe = data->keys_per_pe;
    const int pe = data->pe;
#ifdef USE_CONTEXTS
    const shmemx_ctx_t ctx = data->ctx;
#endif

    for (int i = data->start_key; i < data->end_key; i++) {
        const int j = binary_search(i, keys, keys_per_pe
#ifdef USE_CONTEXTS
                , ctx
#endif
                );
        if (j != i) {
            printf("%2d: Error searching for %d.  Found at index %d, expected "
                    "%d\n", pe, i, j, i);
            data->my_errs += 1;
        }
    }

    return NULL;
}

int main(int argc, char **argv) {
    int i;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <keys-per-pe> <nthreads>\n", argv[0]);
        return 1;
    }

    const int keys_per_pe = atoi(argv[1]);
    const int nthreads = atoi(argv[2]);

#ifndef USE_PTHREADS
    assert(nthreads == 1);
#endif

#ifdef USE_THREAD_INITIALIZER
    int provided;
    shmemx_init_thread(SHMEMX_THREAD_MULTIPLE, &provided);
    assert(provided == SHMEMX_THREAD_MULTIPLE);

#ifdef USE_CONTEXTS
    shmemx_domain_t *domains = (shmemx_domain_t *)malloc(
            nthreads * sizeof(shmemx_domain_t));
    shmemx_ctx_t *contexts = (shmemx_ctx_t *)malloc(
            nthreads * sizeof(shmemx_ctx_t));
    assert(domains && contexts);
    const int derr = shmemx_domain_create(SHMEMX_THREAD_SINGLE, nthreads,
            domains);
    assert(derr == 0);
    for (i = 0; i < nthreads; i++) {
        const int cerr = shmemx_ctx_create(domains[i], &contexts[i]);
        assert(cerr == 0);
    }
#endif
#else
    shmem_init();
#endif

    const int pe = shmem_my_pe();
    const int npes = shmem_n_pes();
    const int total_keys = keys_per_pe * npes;

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
    *my_errs = 0;

    double *my_elapsed_time = (double *)shmem_malloc(sizeof(double));
    double *cumulative_elapsed_time = (double *)shmem_malloc(sizeof(double));
    assert(my_elapsed_time && cumulative_elapsed_time);

    const int keys_per_thread = (total_keys + nthreads - 1) / nthreads;
    thread_context *thread_data = (thread_context *)malloc(
            nthreads * sizeof(thread_context));
    pthread_t *threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    assert(thread_data && threads);
    for (i = 0; i < nthreads; i++) {
        thread_data[i].start_key = i * keys_per_thread;
        thread_data[i].end_key = (i + 1) * keys_per_thread;
        if (thread_data[i].end_key > total_keys) {
            thread_data[i].end_key = total_keys;
        }
        thread_data[i].keys = keys;
        thread_data[i].keys_per_pe = keys_per_pe;
        thread_data[i].pe = pe;
        thread_data[i].my_errs = 0;
#ifdef USE_CONTEXTS
        thread_data[i].ctx = contexts[i];
#endif
    }

    for (i = 0; i < keys_per_pe; i++) {
        keys[i] = keys_per_pe * pe + i;
    }

    shmem_barrier_all();

    const double start_time = shmemx_wtime();

#ifdef USE_PTHREADS
    for (int t = 1; t < nthreads; t++) {
        const int perr = pthread_create(&threads[t], NULL, thread_body,
                &thread_data[t]);
        assert(perr == 0);
    }
#endif

    thread_body(&thread_data[0]);

#ifdef USE_PTHREADS
    for (int t = 1; t < nthreads; t++) {
        const int perr = pthread_join(threads[t], NULL);
        assert(perr == 0);

        *my_errs += thread_data[t].my_errs;
    }
#endif
    *my_errs += thread_data[0].my_errs;

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

#ifdef USE_CONTEXTS
    for (int t = 0; t < nthreads; t++) {
        shmemx_ctx_destroy(contexts[t]);
    }
    shmemx_domain_destroy(nthreads, domains);
#endif
    shmem_finalize();

    return 0;
}
