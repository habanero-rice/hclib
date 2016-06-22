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

/*
 * hclib-forasync.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Ported from hclib
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCLIB_FORASYNC_H_
#define HCLIB_FORASYNC_H_

#include "hclib-place.h"
#include "hclib-cuda.h"

/*
 * Forasync mode to perform static chunking of the iteration space.
 */
#define FORASYNC_MODE_RECURSIVE 1
/*
 * Forasync mode to perform static chunking of the iteration space.
 */
#define FORASYNC_MODE_FLAT 0

/** @struct loop_domain_t
 * @brief Describe loop domain when spawning a forasync.
 * @param[in] low       Lower bound for the loop
 * @param[in] high      Upper bound for the loop
 * @param[in] stride    Stride access
 * @param[in] tile      Tile size for chunking
 */

namespace hclib {

// XXX - This union is a hack to get around the restriction of casting
// function pointers to void* in C++. It would be better if we refactored
// the C API to have 1D, 2D and 3D variants, which take the correctly-typed
// function pointer rather than a void*. We can leave the current C function
// for backward-compatibility, but the explicit 1D/2D/3D versions would be
// the preferred interface going forward.
union for_async_fp_union {
    forasync1D_Fct_t f1;
    forasync2D_Fct_t f2;
    forasync3D_Fct_t f3;
    void *vp;
};

template<typename T>
void forasync1D_wrapper(void *arg, int i) {
    T* lambda = static_cast<T*>(arg);
    (*lambda)(i);
    // FIXME - memory leak
    // we can't delete this here (it's called in a loop)
    // delete lambda;
}

template <typename T>
inline void forasync1D(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    HASSERT(place == NULL || is_cpu_place(place));
    constexpr int DIM = 1;
    // copy lambda
    typedef typename std::remove_reference<T>::type U;
    U *arg = new U(lambda);
    // set up wrapper function
    for_async_fp_union fp;
    fp.f1 = forasync1D_wrapper<U>;
    HASSERT_STATIC(sizeof(fp.f1) == sizeof(fp.vp), "can pass as void*");
    // launch
    if (place || future_list) {
        loop_domain_t domain[] = { loop[0] };
        async_await_at([=]{
                HASSERT_STATIC(sizeof(domain) == sizeof(*domain) * DIM,
                        "The whole domain is captured by value");
                hclib_forasync(fp.vp, arg, nullptr, DIM, domain, mode);
            }, place, future_list);
    }
    else {
        hclib_forasync(fp.vp, arg, nullptr, DIM, loop, mode);
    }
}

template<typename T>
void forasync2D_wrapper(void *arg, int i, int j) {
    T* lambda = static_cast<T*>(arg);
    (*lambda)(i, j);
    // FIXME - memory leak
    // we can't delete this here (it's called in a loop)
    // delete lambda;
}

template <typename T>
inline void forasync2D(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    HASSERT(place == NULL || is_cpu_place(place));
    constexpr int DIM = 2;
    // copy lambda
    typedef typename std::remove_reference<T>::type U;
    U *arg = new U(lambda);
    // set up wrapper function
    for_async_fp_union fp;
    fp.f2 = forasync2D_wrapper<U>;
    HASSERT_STATIC(sizeof(fp.f2) == sizeof(fp.vp), "can pass as void*");
    // launch
    if (place || future_list) {
        loop_domain_t domain[] = { loop[0], loop[1] };
        async_await_at([=]{
                HASSERT_STATIC(sizeof(domain) == sizeof(*domain) * DIM,
                        "The whole domain is captured by value");
                hclib_forasync(fp.vp, arg, nullptr, DIM, domain, mode);
            }, place, future_list);
    }
    else {
        hclib_forasync(fp.vp, arg, nullptr, DIM, loop, mode);
    }
}

template<typename T>
void forasync3D_wrapper(void *arg, int i, int j, int k) {
    T* lambda = static_cast<T*>(arg);
    (*lambda)(i, j, k);
    // FIXME - memory leak
    // we can't delete this here (it's called in a loop)
    // delete lambda;
}

template <typename T>
inline void forasync3D(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    HASSERT(place == NULL || is_cpu_place(place));
    constexpr int DIM = 3;
    // copy lambda
    typedef typename std::remove_reference<T>::type U;
    U *arg = new U(lambda);
    // set up wrapper function
    for_async_fp_union fp;
    fp.f3 = forasync3D_wrapper<U>;
    HASSERT_STATIC(sizeof(fp.f3) == sizeof(fp.vp), "can pass as void*");
    // launch
    if (place || future_list) {
        loop_domain_t domain[] = { loop[0], loop[1], loop[2] };
        async_await_at([=]{
                HASSERT_STATIC(sizeof(domain) == sizeof(*domain) * DIM,
                        "the whole domain is captured by value");
                hclib_forasync(fp.vp, arg, nullptr, DIM, domain, mode);
            }, place, future_list);
    }
    else {
        hclib_forasync(fp.vp, arg, nullptr, DIM, loop, mode);
    }
}


template <typename T>
inline hclib::future_t *forasync1D_future(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    hclib_start_finish();
    forasync1D(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t *event = new hclib::promise_t();
    hclib_end_finish_nonblocking_helper(&event->internal);
    return event->get_future();
}

template <typename T>
inline hclib::future_t *forasync2D_future(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    hclib_start_finish();
    forasync2D(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t *event = new hclib::promise_t();
    hclib_end_finish_nonblocking_helper(&event->internal);
    return event->get_future();
}

template <typename T>
inline hclib::future_t *forasync3D_future(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    hclib_start_finish();
    forasync3D(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t *event = new hclib::promise_t();
    hclib_end_finish_nonblocking_helper(&event->internal);
    return event->get_future();
}

// TODO - Do we also need parameter-pack versions of the other forasyncs?
template <typename T, typename... future_list_t>
inline hclib::future_t *forasync1D_future(loop_domain_t* loop, T &&lambda,
        int mode, place_t *place, future_list_t... futures) {
    // FIXME - memory leak
    hclib_future_t **future_list = construct_future_list(futures...);
    return forasync1D_future(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
}

}

#endif /* HCLIB_FORASYNC_H_ */
