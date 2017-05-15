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

#include <functional>
#include <type_traits>

#include "hclib-async-struct.h"
#include "hclib-promise.hpp"

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
    MARK_BUSY(get_current_worker());
    (*lambda)(); // !!! May cause a worker-swap !!!
    MARK_OVH(get_current_worker());
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
    MARK_BUSY(get_current_worker());
    (*arg->lambda)(); // !!! May cause a worker-swap !!!
    MARK_OVH(get_current_worker());
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
        MARK_BUSY(get_current_worker());
        R res = (*arg->lambda)(); // !!! May cause a worker-swap !!!
        MARK_OVH(get_current_worker());
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
        MARK_BUSY(get_current_worker());
        (*arg->lambda)(); // !!! May cause a worker-swap !!!
        MARK_OVH(get_current_worker());
        arg->event->put();
        delete arg->lambda;
        delete arg;
    }
};

template <typename T>
inline void async(T &&lambda) {
    MARK_OVH(get_current_worker());
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), nullptr, nullptr, nullptr, 0);
}

template <typename T>
inline void async_at_hpt(place_t* pl, T &&lambda) {
    MARK_OVH(get_current_worker());
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), nullptr, nullptr, pl, 0);
}

template <typename T>
inline void async_await(T &&lambda, hclib_future_t **fs) {
    MARK_OVH(get_current_worker());
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
    MARK_OVH(get_current_worker());
    typedef typename std::remove_reference<T>::type U;
    hclib_future_t **fs = construct_future_list(futures...);
    lambda_await_args<U> *args = new lambda_await_args<U> { new U(lambda), fs };
    hclib_async(lambda_await_wrapper<U>, args,
            fs, nullptr, nullptr, 0);
}

template <typename T>
inline void async_await_at(T &&lambda, place_t *pl, hclib_future_t **fs) {
    MARK_OVH(get_current_worker());
    typedef typename std::remove_reference<T>::type U;
    hclib_async(lambda_wrapper<U>, new U(lambda), fs, nullptr, pl, 0);
}

template <typename T, typename... future_list_t>
inline void async_await_at(T &&lambda, place_t *pl, future_list_t... futures) {
    MARK_OVH(get_current_worker());
    typedef typename std::remove_reference<T>::type U;
    hclib_future_t **fs = construct_future_list(futures...);
    lambda_await_args<U> *args = new lambda_await_args<U> { new U(lambda), fs };
    hclib_async(lambda_await_wrapper<U>, args,
            fs, nullptr, pl, 0);
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
