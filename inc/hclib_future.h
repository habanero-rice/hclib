#ifndef HCLIB_FUTURE_H
#define HCLIB_FUTURE_H

#include "hclib-promise.h"

namespace hclib {

struct future_t: public hclib_future_t {
    void *get() {
        return hclib_future_get(this);
    }

    void *wait() {
        return hclib_future_wait(this);
    }
};

HASSERT_STATIC(std::is_pod<future_t>::value, "future_t is plain-old-datatype");
// assert that we can safely cast back and forth between the C and C++ types
HASSERT_STATIC(sizeof(future_t) == sizeof(hclib_future_t),
        "future_t is a trivial wrapper around hclib_future_t");

}

#endif
