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

#include "hclib.h"
#include "hclib-async-struct.h"
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

/*
 * At the lowest layer in the call stack before entering user code, this method
 * invokes the user-provided lambda.
 */
template <typename T>
inline void call_lambda(T* lambda) {
	const int wid = current_ws()->id;
	MARK_BUSY(wid);
	(*lambda)();
	HC_FREE((void*) lambda);
	MARK_OVH(wid);
}

/*
 * Store a reference to the type-specific function for calling the user lambda,
 * as well as a pointer to the lambda's location on the heap (through which we
 * can invoke it). async_arguments is stored as the args field in the task_t
 * object for a task, and passed to lambda_wrapper.
 */
template <typename Function, typename T1>
struct async_arguments {
    Function lambda_caller;
    T1 lambda_on_heap;

    async_arguments(Function k, T1 a) :
        lambda_caller(k), lambda_on_heap(a) { }
};

/*
 * The method called directly from the HC runtime, passed a pointer to an
 * async_arguments object. It then uses these async_arguments to call
 * call_lambda, passing the user-provided lambda.
 */
template<typename Function, typename T1>
void lambda_wrapper(void *args) {
    async_arguments<Function, T1> *a =
        (async_arguments<Function, T1> *)args;

    (*a->lambda_caller)(a->lambda_on_heap);
}

/*
 * Initialize a task_t for the C++ APIs, using a user-provided lambda.
 */
template<typename Function, typename T1>
inline void initialize_task(hclib_task_t *t, Function lambda_caller,
        T1 *lambda_on_heap) {
    async_arguments<Function, T1 *> *args =
        new async_arguments<Function, T1*>(lambda_caller, lambda_on_heap);
    t->_fp = lambda_wrapper<Function, T1 *>;
    t->args = args;
}

template <typename T>
inline hclib_task_t* _allocate_async(T lambda, const bool await) {
	const size_t hclib_task_size = (await ? sizeof(hclib_dependent_task_t) :
            sizeof(hclib_task_t));
    // create off-stack storage for this task
	hclib_task_t* task = (hclib_task_t*)calloc(1, hclib_task_size);
    /*
     * create off-stack storage for the lambda object (including its captured
     * variables), which will be pointed to from the task_t.
     */
	T* lambda_on_heap = (T*)malloc(sizeof(T));
	memcpy(lambda_on_heap, &lambda, sizeof(T));

    initialize_task(task, call_lambda<T>, lambda_on_heap);
	return task;
}

template <typename T>
inline void async(T lambda) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async<T>(lambda, false);
	spawn(task);
}

template <typename T>
inline void async_at(T lambda, hclib_locale_t *locale) {
    MARK_OVH(current_ws()->id);
    hclib_task_t* task = _allocate_async<T>(lambda, false);
    spawn_at(task, locale);
}

template <typename T>
inline void async_nb(T lambda) {
	MARK_OVH(current_ws()->id);
    hclib_task_t *task = _allocate_async<T>(lambda, false);
    task->non_blocking = 1;
	spawn(task);
}

template <typename T>
inline void async_nb_at(T lambda, hclib_locale_t *locale) {
	MARK_OVH(current_ws()->id);
    hclib_task_t *task = _allocate_async<T>(lambda, false);
    task->non_blocking = 1;
	spawn_at(task, locale);
}

template <typename T>
inline void async_nb_await(T lambda, hclib::future_t *future) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async<T>(lambda, true);
    task->non_blocking = 1;
	spawn_await(task, future ? future->internal : NULL, NULL);
}

template <typename T>
inline void async_nb_await_at(T lambda, hclib::future_t *fut,
        hclib_locale_t *locale) {
    MARK_OVH(current_ws()->id);
    hclib_task_t *task = _allocate_async<T>(lambda, true);
    task->non_blocking = 1;
    spawn_await_at(task, fut ? fut->internal : NULL, NULL, locale);
}

inline int _count_futures() {
    return 0;
}
template <typename... future_list_t>
inline int _count_futures(hclib::future_t *future,
        future_list_t... futures) {
    return 1 + _count_futures(futures...);
}
template <typename... future_list_t>
inline int count_futures(future_list_t... futures) {
    return _count_futures(futures...);
}

inline void _construct_future_list(int index, hclib_future_t **future_list,
        hclib::future_t *future) {
    future_list[index] = future->internal;
}
template <typename... future_list_t>
inline void _construct_future_list(int index, hclib_future_t **future_list,
        hclib::future_t *future, future_list_t... remaining) {
    future_list[index] = future->internal;
    _construct_future_list(index + 1, future_list, remaining...);
}

template <typename... future_list_t>
inline hclib_future_t **construct_future_list(future_list_t... futures) {
    const int nfutures = count_futures(futures...);
    hclib_future_t **future_list = (hclib_future_t **)malloc(
            (nfutures + 1) * sizeof(hclib_future_t *));
    HASSERT(future_list);
    _construct_future_list(0, future_list, futures...);
    future_list[nfutures] = NULL;
    return future_list;
}

template <typename T>
inline void async_await(T lambda, hclib::future_t *future) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async<T>(lambda, true);
	spawn_await(task, future ? future->internal : NULL, NULL);
}

template <typename T>
inline void async_await(T lambda, hclib::future_t *future1,
        hclib::future_t *future2) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async<T>(lambda, true);
	spawn_await(task, future1 ? future1->internal : NULL,
            future2 ? future2->internal : NULL);
}

template <typename T>
inline void async_await_at(T lambda, hclib::future_t *future,
        hclib_locale_t *locale) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async<T>(lambda, true);
	spawn_await_at(task, future ? future->internal : NULL, NULL, locale);
}

template <typename T>
inline void async_await_at(T lambda, hclib::future_t *future1,
        hclib::future_t *future2, hclib_locale_t *locale) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async<T>(lambda, true);
	spawn_await_at(task, future1 ? future1->internal : NULL,
            future2 ? future2->internal : NULL, locale);
}


template <typename T>
hclib::future_t *async_future(T lambda) {
    hclib::promise_t *event = new hclib::promise_t();
    hclib_promise_t *internal_event = &event->internal;
    /*
     * TODO creating this closure may be inefficient. While the capture list is
     * precise, if the user-provided lambda is large then copying it by value
     * will also take extra time.
     */
    auto wrapper = [internal_event, lambda]() {
        lambda();
        hclib_promise_put(internal_event, NULL);
    };
    hclib_task_t* task = _allocate_async(wrapper, false);
    spawn(task);
    return event->get_future();
}

template <typename T>
hclib::future_t *async_future_await(T lambda, hclib::future_t *future) {
    hclib::promise_t *event = new hclib::promise_t();
    hclib_promise_t *internal_event = &event->internal;
    /*
     * TODO creating this closure may be inefficient. While the capture list is
     * precise, if the user-provided lambda is large then copying it by value
     * will also take extra time.
     */
    auto wrapper = [internal_event, lambda]() {
        lambda();
        hclib_promise_put(internal_event, NULL);
    };

    hclib_task_t* task = _allocate_async(wrapper, true);
    spawn_await(task, future->internal, NULL);
    return event->get_future();
}

template <typename T>
hclib::future_t *async_future_at_helper(T lambda, hclib_locale_t *locale,
        bool nb) {
    hclib::promise_t *event = new hclib::promise_t();
    hclib_promise_t *internal_event = &event->internal;
    /*
     * TODO creating this closure may be inefficient. While the capture list is
     * precise, if the user-provided lambda is large then copying it by value
     * will also take extra time.
     */
    auto wrapper = [internal_event, lambda]() {
        lambda();
        hclib_promise_put(internal_event, NULL);
    };

    hclib_task_t* task = _allocate_async(wrapper, true);
    if (nb) task->non_blocking = 1;
    spawn_await_at(task, NULL, NULL, locale);
    return event->get_future();
}

template <typename T>
hclib::future_t *async_future_at(T lambda, hclib_locale_t *locale) {
    return async_future_at_helper<T>(lambda, locale, false);
}

template <typename T>
hclib::future_t *async_nb_future_at(T lambda, hclib_locale_t *locale) {
    return async_future_at_helper<T>(lambda, locale, true);
}

template <typename T>
hclib::future_t *async_future_await_at(T lambda, hclib::future_t *future,
        hclib_locale_t *locale) {
    hclib::promise_t *event = new hclib::promise_t();
    hclib_promise_t *internal_event = &event->internal;
    /*
     * TODO creating this closure may be inefficient. While the capture list is
     * precise, if the user-provided lambda is large then copying it by value
     * will also take extra time.
     */
    auto wrapper = [internal_event, lambda]() {
        lambda();
        hclib_promise_put(internal_event, NULL);
    };

    hclib_task_t* task = _allocate_async(wrapper, true);
    spawn_await_at(task, future->internal, NULL, locale);
    return event->get_future();
}

inline void finish(std::function<void()> lambda) {
    hclib_start_finish();
    lambda();
    hclib_end_finish();
}

inline hclib::future_t *nonblocking_finish(std::function<void()> lambda) {
    hclib_start_finish();
    lambda();
    hclib::promise_t *event = new hclib::promise_t();
    hclib_end_finish_nonblocking_helper(&event->internal);
    return event->get_future();
}

inline void yield() {
    hclib_yield(NULL);
}

inline void yield_at(hclib_locale_t *locale) {
    hclib_yield(locale);
}

}

#endif /* HCLIB_ASYNC_H_ */
