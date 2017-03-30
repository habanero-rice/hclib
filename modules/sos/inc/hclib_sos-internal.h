#ifndef HCLIB_OPENSHMEM_INTERNAL_H
#define HCLIB_OPENSHMEM_INTERNAL_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"
#include "hclib-module-common.h"

#include <shmem.h>

#include <stdlib.h>
#include <stdio.h>

enum wait_type {
    integer
};

typedef struct _pending_sos_op {
    wait_type type;
    volatile void *var;
    int cmp;
    union { int i; } cmp_value;

    hclib::promise_t<void> *prom;
    hclib_task_t *task;

    struct _pending_sos_op *next;
#ifdef HCLIB_INSTRUMENT
    int event_type;
    int event_id;
#endif
} pending_sos_op;

extern pending_sos_op *pending;
extern bool test_sos_completion(void *generic_op);
extern hclib::locale_t *nic;

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_post_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_finalize);

int shmem_my_pe();
int shmem_n_pes();
void *shmem_malloc(size_t size);
void shmem_free(void *ptr);
void shmem_barrier_all();
void shmem_fence();
void shmem_quiet();
void shmem_put64(void *dest, const void *source, size_t nelems, int pe);
void shmem_broadcast64(void *dest, const void *source, size_t nelems,
        int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);

void shmem_set_lock(volatile long *lock);
void shmem_clear_lock(long *lock);

void shmem_int_get(int *dest, const int *source, size_t nelems, int pe);
void shmem_int_put(int *dest, const int *source, size_t nelems, int pe);
void shmem_longlong_put(long long *dest, const long long *src,
                        size_t nelems, int pe);
void shmem_getmem(void *dest, const void *source, size_t nelems, int pe);
void shmem_putmem(void *dest, const void *source, size_t nelems, int pe);

void shmem_char_put_nbi(char *dest, const char *source, size_t nelems, int pe);
void shmem_char_put_signal_nbi(char *dest, const char *source,
        size_t nelems, char *signal_dest, const char *signal_source,
        size_t signal_nelems, int pe);

void shmem_longlong_p(long long *addr, long long value, int pe);

void shmem_int_add(int *dest, int value, int pe);
int shmem_int_fadd(int *dest, int value, int pe);
int shmem_int_finc(int *dest, int pe);
long shmem_long_finc(long *dest, int pe);
int shmem_int_cswap(int *dest, int cond, int value, int pe);
int shmem_int_swap(int *dest, int value, int pe);
int shmem_int_fetch(const int *dest, int pe);
long long shmem_longlong_fadd(long long *target, long long value,
                              int pe);

void shmem_int_sum_to_all(int *target, int *source, int nreduce,
                          int PE_start, int logPE_stride,
                          int PE_size, int *pWrk, long *pSync);
void shmem_longlong_sum_to_all(long long *target, long long *source,
                               int nreduce, int PE_start,
                               int logPE_stride, int PE_size,
                               long long *pWrk, long *pSync);
void shmem_longlong_max_to_all(long long *target, long long *source,
                               int nreduce, int PE_start,
                               int logPE_stride, int PE_size,
                               long long *pWrk, long *pSync);

void shmem_collect32(void *dest, const void *source, size_t nelems,
        int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_fcollect64(void *dest, const void *source, size_t nelems,
        int PE_start, int logPE_stride, int PE_size, long *pSync);

void shmem_int_wait_until(volatile int *ivar, int cmp, int cmp_value);

template <typename T>
void shmem_int_async_when(volatile int *ivar, int cmp,
        int cmp_value, T&& lambda) {
    typedef typename std::remove_reference<T>::type U;

    pending_sos_op *op = (pending_sos_op *)malloc(sizeof(*op));
    assert(op);

    op->type = integer;
    op->var = ivar;
    op->cmp = cmp;
    op->cmp_value.i = cmp_value;
    op->prom = NULL;
    op->task = _allocate_async(new U(lambda));
#ifdef HCLIB_INSTRUMENT
    op->event_type = event_ids[shmem_int_async_nb_when_lbl];
    op->event_id = _event_id;
#endif
    hclib::append_to_pending(op, &pending, test_sos_completion, nic);
}

template <typename T>
void shmem_int_async_nb_when(volatile int *ivar, int cmp,
        int cmp_value, T&& lambda) {
    typedef typename std::remove_reference<T>::type U;

    pending_sos_op *op = (pending_sos_op *)malloc(sizeof(*op));
    assert(op);

    op->type = integer;
    op->var = ivar;
    op->cmp = cmp;
    op->cmp_value.i = cmp_value;
    op->prom = NULL;
    op->task = _allocate_async(new U(lambda));
    op->task->non_blocking = 1;
#ifdef HCLIB_INSTRUMENT
    op->event_type = event_ids[shmem_int_async_nb_when_lbl];
    op->event_id = _event_id;
#endif
    hclib::append_to_pending(op, &pending, test_sos_completion, nic);
}

std::string shmem_name();

}

#endif
