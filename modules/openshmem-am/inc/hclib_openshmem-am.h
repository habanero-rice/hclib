#ifndef HCLIB_OPENSHMEM_AM_H
#define HCLIB_OPENSHMEM_AM_H

#include "hclib_openshmem-am-internal.h"

extern int handler_func_id;

namespace hclib {

template <class T>
void caller(void *lambda) {
    T *unwrapped = (T *)lambda;
    (*unwrapped)();
}

template <class T>
void async_remote(T lambda, const int pe) {
#ifdef VERBOSE
    fprintf(stderr, "async_remote: fp = %p , pe = %d , source PE = %d\n", fp,
            pe, ::shmem_my_pe());
#endif

    void (*fp)(void *) = caller<T>;

    char *buf = (char *)malloc(sizeof(void *) + sizeof(lambda));
    assert(buf);
    memcpy(buf, &fp, sizeof(fp));
    memcpy(buf + sizeof(fp), &lambda, sizeof(lambda));

    shmemx_am_request(pe, handler_func_id, buf, sizeof(void *) + sizeof(lambda));
    shmemx_am_quiet();
}

}

#endif
