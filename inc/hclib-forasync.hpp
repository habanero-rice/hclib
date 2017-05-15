/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HCLIB_FORASYNC_H_
#define HCLIB_FORASYNC_H_

#include "hclib-place.h"

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
inline hclib::future_t<void> *forasync1D_future(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    hclib_start_finish();
    forasync1D(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t<void> *event = new hclib::promise_t<void>();
    hclib_end_finish_nonblocking_helper(event);
    return event->get_future();
}

template <typename T>
inline hclib::future_t<void> *forasync2D_future(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    hclib_start_finish();
    forasync2D(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t<void> *event = new hclib::promise_t<void>();
    hclib_end_finish_nonblocking_helper(event);
    return event->get_future();
}

template <typename T>
inline hclib::future_t<void> *forasync3D_future(loop_domain_t* loop, T &&lambda,
        int mode = FORASYNC_MODE_RECURSIVE, place_t *place = NULL,
        hclib_future_t **future_list = NULL) {
    hclib_start_finish();
    forasync3D(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t<void> *event = new hclib::promise_t<void>();
    hclib_end_finish_nonblocking_helper(event);
    return event->get_future();
}

// TODO - Do we also need parameter-pack versions of the other forasyncs?
template <typename T, typename... future_list_t>
inline hclib::future_t<void> *forasync1D_future(loop_domain_t* loop, T &&lambda,
        int mode, place_t *place, future_list_t... futures) {
    // FIXME - memory leak
    hclib_future_t **future_list = construct_future_list(futures...);
    return forasync1D_future(loop, std::forward<decltype(lambda)>(lambda),
            mode, place, future_list);
}

}

#endif /* HCLIB_FORASYNC_H_ */
