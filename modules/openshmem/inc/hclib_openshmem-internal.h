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
void shmem_barrier_all();
void shmem_put64(void *dest, const void *source, size_t nelems, int pe);
void shmem_broadcast64(void *dest, const void *source, size_t nelems,
        int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);

locale_t *shmem_remote_pe(int pe);
int pe_for_locale(locale_t *locale);

}

#endif
