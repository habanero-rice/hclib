#ifndef HCLIB_PROMISE_H
#define HCLIB_PROMISE_H

#include "hclib-promise.h"
#include "hclib_future.h"

namespace hclib {

struct promise_t: public hclib_promise_t {
    promise_t() {
        hclib_promise_init(this);
    }
    ~promise_t() { }

    void put(void *datum) {
        hclib_promise_put(this, datum);
    }

    future_t *get_future() {
        return static_cast<future_t*>(&this->future);
    }
};

// assert that we can safely cast back and forth between the C and C++ types
HASSERT_STATIC(sizeof(promise_t) == sizeof(hclib_promise_t),
        "promise_t is a trivial wrapper around hclib_promise_t");

}

#endif
