#ifndef HCLIB_OPENSHMEM_INTERNAL_H
#define HCLIB_OPENSHMEM_INTERNAL_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#include <shmem.h>

#include <stdlib.h>
#include <stdio.h>

namespace hclib {

enum wait_type {
    integer
};

typedef union _wait_cmp_value_t {
    int i;
} wait_cmp_value_t;

typedef struct _wait_info_t {
    wait_type type;
    volatile void *var;
    int cmp;
    wait_cmp_value_t cmp_value;
} wait_info_t;

typedef struct _wait_set_t {
    wait_info_t *infos;
    int ninfos;
    hclib_promise_t *signal;
    hclib_task_t *task;

    struct _wait_set_t *next;
} wait_set_t;

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
void shmem_int_wait_until_any(volatile int **ivars, int cmp,
        int *cmp_values, int nwaits);

void reset_oshmem_profiling_data();
void print_oshmem_profiling_data();
void enable_oshmem_profiling();
void disable_oshmem_profiling();

#define construct_and_insert_wait_set(vars, cmp, cmp_values, nwaits, wait_type, fieldname, dependent_task) ({ \
    wait_info_t *infos = (wait_info_t *)malloc((nwaits) * sizeof(wait_info_t)); \
    HASSERT(infos); \
    for (int i = 0; i < (nwaits); i++) { \
        infos[i].type = (wait_type); \
        infos[i].var = (vars)[i]; \
        infos[i].cmp = (cmp); \
        infos[i].cmp_value.fieldname = (cmp_values)[i]; \
    } \
    \
    wait_set_t *wait_set = (wait_set_t *)calloc(1, sizeof(wait_set_t)); \
    HASSERT(wait_set); \
    \
    if (dependent_task) wait_set->task = (dependent_task); \
    else wait_set->signal = hclib_promise_create(); \
    \
    wait_set->ninfos = (nwaits); \
    wait_set->infos = infos; \
    \
    enqueue_wait_set(wait_set); \
    \
    wait_set->signal; \
})

void enqueue_wait_set(wait_set_t *wait_set);

template <typename T>
void shmem_int_async_when(volatile int *ivar, int cmp,
        int cmp_value, T&& lambda) {
    typedef typename std::remove_reference<T>::type U;
    hclib_task_t *task = _allocate_async(new U(lambda));

    hclib_promise_t *promise = construct_and_insert_wait_set(&ivar, cmp,
            &cmp_value, 1, integer, i, task);
    HASSERT(promise == NULL);
}

template <typename T>
void shmem_int_async_nb_when(volatile int *ivar, int cmp,
        int cmp_value, T&& lambda) {
    typedef typename std::remove_reference<T>::type U;
    hclib_task_t *task = _allocate_async(new U(lambda));
    task->non_blocking = 1;

    hclib_promise_t *promise = construct_and_insert_wait_set(&ivar, cmp,
            &cmp_value, 1, integer, i, task);
    HASSERT(promise == NULL);
}

template <typename T>
void shmem_int_async_when_any(volatile int **ivars, int cmp,
        int *cmp_values, int nwaits, T&& lambda) {
    typedef typename std::remove_reference<T>::type U;
    hclib_task_t *task = _allocate_async(new U(lambda));

    hclib_promise_t *promise = construct_and_insert_wait_set(ivars, cmp,
            cmp_values, nwaits, integer, i, task);
    HASSERT(promise == NULL);
}

std::string shmem_name();

}

#endif
