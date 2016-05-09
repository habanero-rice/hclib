#ifndef HCLIB_FINISH_H
#define HCLIB_FINISH_H

#include "hclib-promise.h"

typedef struct finish_t {
    struct finish_t* parent;
    volatile int counter;
#ifdef _HC_MASTER_OWN_MAIN_FUNC_
    int outtermost_user_end_finish;
#endif
#if HCLIB_LITECTX_STRATEGY
    hclib_future_t ** finish_deps;
#endif /* HCLIB_LITECTX_STRATEGY */
} finish_t;

#endif
