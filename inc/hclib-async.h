/* Copyright (c) 2016, Rice University

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
 * hclib-async.h
 *  
 *      Authors: Vivek Kumar (vivekk@rice.edu), Max Grossman (jmg3@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */
#include <functional>
#include <type_traits>

#include "hclib-async-struct.h"
#include "hcupc-support.h"
#include "hclib_promise.h"

#ifndef HCLIB_ASYNC_H_
#define HCLIB_ASYNC_H_

namespace hclib {

/*
 * The C API to the HC runtime defines a task at its simplest as a function
 * pointer paired with a void* pointing to some user data. This file adds a C++
 * wrapper over that API by passing the C API a lambda-caller function and a
 * pointer to the lambda stored on the heap, which are then called.
 *
 * This does add more overheads in the C++ version (i.e. memory allocations).
 * TODO optimize that overhead.
 */

/* raw function pointer for calling lambdas */
template<typename T>
void lambda_wrapper(void *arg) {
    T *lambda = static_cast<T*>(arg);
    MARK_BUSY(current_ws()->id);
    (*lambda)(); // !!! May cause a worker-swap !!!
    MARK_OVH(current_ws()->id);
    delete lambda;
}


/* this version also deletes a runtime-managed future list */
template<typename T>
struct lambda_await_args {
    T* lambda;
    hclib_future_t **future_list;
};
template<typename T>
void lambda_await_wrapper(void *raw_arg) {
    auto arg = static_cast<lambda_await_args<T>*>(raw_arg);
    delete[] arg->future_list;
    MARK_BUSY(current_ws()->id);
    (*arg->lambda)(); // !!! May cause a worker-swap !!!
    MARK_OVH(current_ws()->id);
    delete arg->lambda;
    delete arg;
}

/* this version also puts the result of the lambda into a promise */
template<typename T, typename R>
struct LambdaFutureArgs {
    T *lambda;
    promise_t<R> *event;
    hclib_future_t **future_list;
};
// NOTE: C++11 does not allow partial specialization of function templates,
// so instead we have to do this awkward thing with static methods.
template<typename T, typename R>
struct LambdaFutureWrapper {
    static void fn(void *raw_arg) {
        auto arg = static_cast<LambdaFutureArgs<T,R>*>(raw_arg);
        delete[] arg->future_list;
        MARK_BUSY(current_ws()->id);
        R res = (*arg->lambda)(); // !!! May cause a worker-swap !!!
        MARK_OVH(current_ws()->id);
        arg->event->put(res);
        delete arg->lambda;
        delete arg;
    }
};
template<typename T>
struct LambdaFutureWrapper<T, void> {
    static void fn(void *raw_arg) {
        auto arg = static_cast<LambdaFutureArgs<T, void>*>(raw_arg);
        delete[] arg->future_list;
        MARK_BUSY(current_ws()->id);
        (*arg->lambda)(); // !!! May cause a worker-swap !!!
        MARK_OVH(current_ws()->id);
        arg->event->put();
        delete arg->lambda;
        delete arg;
    }
};

/*
 * Yes, the name "async_at_hpt" sounds weird
 * but using this name to differentiate with the inter-node
 * "asyncAt" in HabaneroUPC++. Feel free to give a better
 * name to async_at_hpt.
 */
template <typename T>
inline void async(T &&lambda) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), nullptr, nullptr, nullptr, 0);
}

template <typename T>
inline void async_at_hpt(place_t* pl, T &&lambda) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), nullptr, nullptr, pl, 0);
}

template <typename T>
inline void async_await(T &&lambda, hclib_future_t **fs) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), fs, nullptr, nullptr, 0);
}

template <typename... Ts>
inline hclib_future_t **construct_future_list(Ts... futures) {
    const size_t n = sizeof...(futures); // parameter pack count
    return new hclib_future_t*[n+1] { futures..., nullptr };
}

template <typename T, typename... future_list_t>
inline void async_await(T &&lambda, future_list_t... futures) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_future_t **fs = construct_future_list(futures...);
    lambda_await_args<U> *args = new lambda_await_args<U> { new U(lambda), fs };
    hclib_async(lambda_await_wrapper<U>, args,
            fs, nullptr, nullptr, 0);
}

template <typename T>
inline void async_await_at(T &&lambda, place_t *pl, hclib_future_t **fs) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), fs, nullptr, pl, 0);
}

template <typename T, typename... future_list_t>
inline void async_await_at(T &&lambda, place_t *pl, future_list_t... futures) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_future_t **fs = construct_future_list(futures...);
    lambda_await_args<U> *args = new lambda_await_args<U> { new U(lambda), fs };
    hclib_async(lambda_await_wrapper<U>, args,
            fs, nullptr, pl, 0);
}

template <typename T>
inline void async_comm(T &&lambda) {
    MARK_OVH(current_ws()->id);
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda),
            nullptr, nullptr, nullptr, COMM_ASYNC);
}

template <typename T>
auto async_future(T &&lambda) -> hclib::future_t<decltype(lambda())>* {
    typedef decltype(lambda()) R;
    typedef typename std::remove_reference<T>::type U;
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t<R> *event = new hclib::promise_t<R>();
    auto args = new LambdaFutureArgs<U,R> { new U(lambda), event, nullptr };
    hclib_async(LambdaFutureWrapper<U,R>::fn, args, nullptr, nullptr, nullptr, 0);
    return event->get_future();
}

template <typename T, typename... future_list_t>
auto async_future_await(T &&lambda, future_list_t... futures) -> hclib::future_t<decltype(lambda())>* {
    typedef decltype(lambda()) R;
    typedef typename std::remove_reference<T>::type U;
    // FIXME - memory leak? (no handle to destroy the promise)
    hclib::promise_t<R> *event = new hclib::promise_t<R>();
    hclib_future_t **fs = construct_future_list(futures...);
    auto args = new LambdaFutureArgs<U,R> { new U(lambda), event, fs };
    hclib_async(LambdaFutureWrapper<U,R>::fn, args, fs, nullptr, nullptr, 0);
    return event->get_future();
}

template <typename T>
inline void finish(T &&lambda) {
    hclib_start_finish();
    lambda();
    hclib_end_finish();
}

template <typename T>
inline hclib::future_t<void> *nonblocking_finish(T &&lambda) {
    hclib_start_finish();
    lambda();
    hclib::promise_t<void> *event = new hclib::promise_t<void>();
    hclib_end_finish_nonblocking_helper(event);
    return event->get_future();
}

}

#endif /* HCLIB_ASYNC_H_ */
