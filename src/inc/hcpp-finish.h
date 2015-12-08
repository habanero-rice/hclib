#ifndef HCPP_FINISH_H
#define HCPP_FINISH_H

#include "hcpp-ddf.h"

typedef struct finish_t {
    struct finish_t* parent;
    volatile int counter;
#if HCLIB_LITECTX_STRATEGY
    hclib_ddf_t ** finish_deps;
#endif /* HCLIB_LITECTX_STRATEGY */
} finish_t;

#endif
