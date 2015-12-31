#ifndef HCLIB_PROMISE_H
#define HCLIB_PROMISE_H

#include "hclib-promise.h"

namespace hclib {

class promise_t {
    public:
        hclib_promise_t internal;

        promise_t() {
            hclib_promise_init(&internal);
        }
        ~promise_t() { }

        void put(void *datum) {
            hclib_promise_put(&internal, datum);
        }
        void *get() {
            return hclib_promise_get(&internal);
        }
        void *wait() {
            return hclib_promise_wait(&internal);
        }
};

}

#endif
