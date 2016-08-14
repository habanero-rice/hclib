#ifndef HCLIB_FUTURE_H
#define HCLIB_FUTURE_H

#include "hclib-promise.h"

namespace hclib {

class future_t {
    public:
        hclib_future_t *internal;
        explicit future_t(hclib_future_t *set_internal) :
            internal(set_internal) { }
        ~future_t() { }

        void *get() {
            return hclib_future_get(internal);
        }

        void *wait() {
            return hclib_future_wait(internal);
        }
};

// FIXME - we should be able to make future_t into a zero-overhead wrapper
// around hclib_future_t, possibly via inheritance + static_cast
//HASSERT_STATIC(std::is_pod<future_t>::value);
//HASSERT_STATIC(sizeof(future_t) == sizeof(hclib_future_t));

}

#endif
