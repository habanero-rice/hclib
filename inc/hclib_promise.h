/* Copyright (c) 2015, Rice University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
3.  Neither the name of Rice University
     nor the names of its contributors may be used to endorse or
     promote products derived from this software without specific
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
#if HAVE_CXX11_TRIVIAL_COPY_CHECK
    HASSERT_STATIC(std::is_trivially_copyable<T>::value,
            "promise_t arg type can't be recast to void*.");
#endif  // HAVE_CXX11_TRIVIAL_COPY_CHECK

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


