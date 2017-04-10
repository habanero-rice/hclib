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
void caller_with_user_data(void *lambda, void *user_data, size_t nbytes) {
    T *unwrapped = (T *)lambda;
    (*unwrapped)(user_data, nbytes);
}

template <class T>
void async_remote(T lambda, const int pe) {
#ifdef VERBOSE
    fprintf(stderr, "async_remote: fp = %p , pe = %d , source PE = %d\n", fp,
            pe, ::shmem_my_pe());
#endif

    void (*fp)(void *) = caller<T>;

    char *buf = (char *)malloc(sizeof(am_packet) + sizeof(lambda));
    assert(buf);
    ((am_packet *)buf)->lambda_size = sizeof(lambda);
    ((am_packet *)buf)->user_data_size = 0;
    ((am_packet *)buf)->fp = (void *)fp;
    memcpy(buf + sizeof(am_packet), &lambda, sizeof(lambda));

    shmemx_am_request(pe, handler_func_id, buf,
            sizeof(am_packet) + sizeof(lambda));
    shmemx_am_quiet();
}

template <class T>
void async_remote(T lambda, const int pe, void *user_data, size_t nbytes) {
#ifdef VERBOSE
    fprintf(stderr, "async_remote: fp = %p , pe = %d , source PE = %d\n", fp,
            pe, ::shmem_my_pe());
#endif

    void (*fp)(void *, void *, size_t) = caller_with_user_data<T>;

    char *buf = (char *)malloc(sizeof(am_packet) + sizeof(lambda) + nbytes);
    assert(buf);
    ((am_packet *)buf)->lambda_size = sizeof(lambda);
    ((am_packet *)buf)->user_data_size = nbytes;
    ((am_packet *)buf)->fp = (void *)fp;
    memcpy(buf + sizeof(am_packet), &lambda, sizeof(lambda));
    memcpy(buf + (sizeof(am_packet) + sizeof(lambda)), user_data, nbytes);

    shmemx_am_request(pe, handler_func_id, buf,
            sizeof(am_packet) + sizeof(lambda) + nbytes);
    shmemx_am_quiet();
}

}

#endif
