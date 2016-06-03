#ifndef HCLIB_OPENSHMEM_INTERNAL_H
#define HCLIB_OPENSHMEM_INTERNAL_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#include <shmem.h>

#include <stdlib.h>
#include <stdio.h>

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_post_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_finalize);

locale_t *shmem_my_pe();
int shmem_n_pes();
void *shmem_malloc(size_t size);
void shmem_free(void *ptr);
void shmem_barrier_all();
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

void shmem_longlong_p(long long *addr, long long value, int pe);

void shmem_int_add(int *dest, int value, int pe);
int shmem_int_fadd(int *dest, int value, int pe);
long long shmem_longlong_fadd(long long *target, long long value,
                              int pe);

void shmem_int_sum_to_all(int *target, int *source, int nreduce,
                          int PE_start, int logPE_stride,
                          int PE_size, int *pWrk, long *pSync);
void shmem_longlong_sum_to_all(long long *target, long long *source,
                               int nreduce, int PE_start,
                               int logPE_stride, int PE_size,
                               long long *pWrk, long *pSync);

locale_t *shmem_remote_pe(int pe);
int pe_for_locale(locale_t *locale);
std::string shmem_name();

}

#endif
