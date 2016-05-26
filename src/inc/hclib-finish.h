#ifndef HCLIB_FINISH_H
#define HCLIB_FINISH_H

#include "hclib-promise.h"

typedef struct finish_t {
    struct finish_t* parent;
    volatile int counter;
    hclib_future_t *finish_dep;
} finish_t;

#endif
