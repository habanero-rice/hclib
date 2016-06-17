#ifndef HCLIB_PROMISE_H
#define HCLIB_PROMISE_H

#include "hclib-promise.h"
#include "hclib_future.h"

namespace hclib {

// Specialized for pointers
// (and pointer-sized value types)
template<typename T>
struct promise_t: public hclib_promise_t {

    HASSERT_STATIC(sizeof(T) <= sizeof(void*),
            "promise_t arg type is too large to store by value.");
    HASSERT_STATIC(std::is_trivially_copyable<T>::value,
            "promise_t arg type can't be recast to void*.");

    promise_t() { hclib_promise_init(this); }

    void put(T datum) {
        void *tmp;
        *reinterpret_cast<T*>(&tmp) = datum;
        hclib_promise_put(this, tmp);
    }

    future_t<T> *get_future() {
        // this is the simplest expression I could come up with
        // that makes the compiler happy, since accessing the shadowed
        // &hclib_promise_t::future is treated as address-of-member...
        return static_cast<future_t<T>*>(
                &static_cast<hclib_promise_t*>(this)->future);
    }

    // we want to hide the raw future member of the parent struct
    future_t<T> &future() { return *get_future(); }
};

// Specialized for pointers
template<typename T>
struct promise_t<T*>: public hclib_promise_t {
    promise_t() { hclib_promise_init(this); }

    void put(T *datum) {
        hclib_promise_put(this, datum);
    }

    future_t<T*> *get_future() {
        return static_cast<future_t<T*>*>(
                &static_cast<hclib_promise_t*>(this)->future);
    }

    future_t<T*> &future() { return *get_future(); }
};

// Specialized for references
template<typename T>
struct promise_t<T&>: public hclib_promise_t {
    promise_t() { hclib_promise_init(this); }

    void put(T &datum) {
        hclib_promise_put(this, &datum);
    }

    future_t<T&> *get_future() {
        return static_cast<future_t<T&>*>(
                &static_cast<hclib_promise_t*>(this)->future);
    }

    future_t<T&> &future() { return *get_future(); }
};

// Specialized for void
template<>
struct promise_t<void>: public hclib_promise_t {
    promise_t() { hclib_promise_init(this); }

    void put() {
        hclib_promise_put(this, nullptr);
    }

    future_t<void> *get_future() {
        return static_cast<future_t<void>*>(
                &static_cast<hclib_promise_t*>(this)->future);
    }

    future_t<void> &future() { return *get_future(); }
};

// assert that we can safely cast back and forth between the C and C++ types
HASSERT_STATIC(sizeof(promise_t<void*>) == sizeof(hclib_promise_t),
        "promise_t is a trivial wrapper around hclib_promise_t");

}

#endif


