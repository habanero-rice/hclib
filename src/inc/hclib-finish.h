#ifndef HCLIB_FINISH_H
#define HCLIB_FINISH_H

#include "hclib-promise.h"

typedef struct finish_t {
    struct finish_t* parent;
    volatile int counter;
#if HCLIB_LITECTX_STRATEGY
    hclib_promise_t ** finish_deps;
#endif /* HCLIB_LITECTX_STRATEGY */
} finish_t;

#endif
