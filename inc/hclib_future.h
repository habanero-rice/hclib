#ifndef HCLIB_FUTURE_H
#define HCLIB_FUTURE_H

#include "hclib-promise.h"

namespace hclib {

// Specialized for scalar types
template<typename T>
struct future_t: public hclib_future_t {

    HASSERT_STATIC(sizeof(T) <= sizeof(void*),
            "future_t arg type is too large to store by value.");
#if HAVE_CXX11_TRIVIAL_COPY_CHECK
    HASSERT_STATIC(std::is_trivially_copyable<T>::value,
            "future_t arg type can't be recast to void*.");
#endif  // HAVE_CXX11_TRIVIAL_COPY_CHECK

    union _ValUnion { T val; void *vp; };

    T get() {
        _ValUnion tmp;
        tmp.vp = hclib_future_get(this);
        return tmp.val;
    }

    T wait() {
        _ValUnion tmp;
        tmp.vp = hclib_future_wait(this);
        return tmp.val;
    }

    bool test() {
        return hclib_future_is_satisfied(this);
    }
};

// Specialized for pointers
template<typename T>
struct future_t<T*>: public hclib_future_t {
    T *get() {
        return static_cast<T*>(hclib_future_get(this));
    }

    T *wait() {
        return static_cast<T*>(hclib_future_wait(this));
    }
};

// Specialized for references
template<typename T>
struct future_t<T&>: public hclib_future_t {
    T &get() {
        return *static_cast<T*>(hclib_future_get(this));
    }

    T &wait() {
        return *static_cast<T*>(hclib_future_wait(this));
    }
};

// Specialized for void
template<>
struct future_t<void>: public hclib_future_t {
    void get() { }
    void wait() { hclib_future_wait(this); }
};

#ifndef __CUDACC__
HASSERT_STATIC(std::is_pod<future_t<void*>>::value,
        "future_t is plain-old-datatype");
#endif
// assert that we can safely cast back and forth between the C and C++ types
HASSERT_STATIC(sizeof(future_t<void*>) == sizeof(hclib_future_t),
        "future_t is a trivial wrapper around hclib_future_t");

}

#endif
