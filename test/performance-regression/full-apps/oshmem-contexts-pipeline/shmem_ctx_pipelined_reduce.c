#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <shmemx.h>
#include <assert.h>

#define LEN  16777216 /* Full buffer length */
#define PLEN 4096  /* Length of each pipeline stage */

int in_buf[LEN], out_buf[LEN];

int main(void) {
    int i, j, *pbuf[2];

#ifndef NO_CONTEXTS
    shmemx_domain_t domains[2];
#endif
    shmemx_ctx_t ctx[2];

    int supported;
    shmemx_init_thread(SHMEMX_THREAD_SINGLE, &supported);
    assert(supported == SHMEMX_THREAD_SINGLE);

    const int me = shmem_my_pe();
    const int npes = shmem_n_pes();

    pbuf[0] = shmem_malloc(PLEN * npes * sizeof(int));
    pbuf[1] = shmem_malloc(PLEN * npes * sizeof(int));

#ifdef NO_CONTEXTS
    ctx[0] = SHMEMX_CTX_DEFAULT;
    ctx[1] = SHMEMX_CTX_DEFAULT;
#else
    int err = shmemx_domain_create(SHMEMX_THREAD_SINGLE, 2, domains);
    assert(err == 0);

    int ret_0 = shmemx_ctx_create(domains[0], &ctx[0]);
    int ret_1 = shmemx_ctx_create(domains[1], &ctx[1]);
    if (ret_0 || ret_1) shmem_global_exit(1);
#endif

    for (i = 0; i < LEN; i++) {
        in_buf[i] = me; out_buf[i] = 0;
    }

    shmemx_sync_all();
    const double start_time = shmemx_wtime();

    int p_idx = 0, p = 0; /* Index of ctx and pbuf (p_idx) for current pipeline stage (p) */
    for (i = 1; i <= npes; i++) {
        shmemx_ctx_int_put_nbi(&pbuf[p_idx][PLEN*me],
                &in_buf[PLEN*p], PLEN, (me+i) % npes, ctx[p_idx]);
    }

    /* Issue communication for pipeline stage p, then accumulate results for stage p-1 */
    for (p = 1; p < LEN/PLEN; p++) {
        p_idx ^= 1;
        for (i = 1; i <= npes; i++)
            shmemx_ctx_int_put_nbi(&pbuf[p_idx][PLEN*me],
                    &in_buf[PLEN*p], PLEN, (me+i) % npes, ctx[p_idx]);

        shmemx_ctx_quiet(ctx[p_idx^1]);
        shmemx_sync_all();
        for (i = 0; i < npes; i++)
            for (j = 0; j < PLEN; j++)
                out_buf[PLEN*(p-1)+j] += pbuf[p_idx^1][PLEN*i+j];
    }

    shmemx_ctx_quiet(ctx[p_idx]);
    shmemx_sync_all();

    for (i = 0; i < npes; i++)
        for (j = 0; j < PLEN; j++)
            out_buf[PLEN*(p-1)+j] += pbuf[p_idx][PLEN*i+j];

    shmemx_sync_all();

    const double elapsed_time = shmemx_wtime() - start_time;
    if (me == 0) {
        printf("Elapsed time for LEN = %d and PLEN = %d is %f s\n", LEN, PLEN, elapsed_time);
    }

#ifndef NO_CONTEXTS
    shmemx_ctx_destroy(ctx[0]);
    shmemx_ctx_destroy(ctx[1]);
    shmemx_domain_destroy(2, domains);
#endif

    shmem_finalize();
    return 0;
}
