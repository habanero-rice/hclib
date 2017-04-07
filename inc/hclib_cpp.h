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

#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.h"
#include "hclib-forasync.h"
#include "hclib_promise.h"
#include "hclib.h"

namespace hclib {

typedef loop_domain_t loop_domain_t;
typedef place_t place_t;
typedef place_type_t place_type_t;

template <typename T>
inline void launch(T &&lambda) {
    hclib_launch(lambda_wrapper<T>, new T(lambda));
}

extern hclib_worker_state *current_ws();
int current_worker();
int num_workers();
int get_num_places(place_type_t type);
void get_places(place_t **pls, place_type_t type);
place_t *get_current_place();
place_t **get_children_of_place(place_t *pl, int *num_children);
place_t *get_root_place();
char *get_place_name(place_t *pl);

#ifdef HC_CUDA
place_t **get_nvgpu_places(int *n_nvgpu_places);

template<typename T>
T *allocate_at(place_t *pl, size_t nitems, int flags) {
    return (T *)hclib_allocate_at(pl, nitems * sizeof(T), flags);
}

template<typename T>
void free_at(place_t *pl, T *ptr) {
    return hclib_free_at(pl, ptr);
}

template<typename T>
future_t<void> *async_copy(place_t *dst_pl, T *dst, place_t *src_pl, T *src,
        size_t nitems, void *user_arg) {
    hclib::promise_t<void> *promise = new hclib::promise_t<void>();
    hclib_async_copy_helper(dst_pl, dst, src_pl, src, nitems * sizeof(T),
            NULL, user_arg, &promise->internal);
    return promise->get_future();
}

template<typename T, typename... future_list_t>
future_t<void> *async_copy(place_t *dst_pl, T *dst,
        place_t *src_pl, T *src, size_t nitems,
        void *user_arg, future_list_t... futures) {
    hclib::promise_t<void> *promise = new hclib::promise_t<void>();
    hclib_future_t **future_list = construct_future_list(futures...);
    hclib_async_copy_helper(dst_pl, dst, src_pl, src, nitems * sizeof(T),
            future_list, user_arg, &promise->internal);
    return promise->get_future();
}

template<typename T>
future_t<void> *async_memset(place_t *pl, T *ptr, int val,
        size_t nitems, void *user_arg) {
    hclib::promise_t<void> *promise = new hclib::promise_t<void>();
    hclib_async_memset_helper(pl, ptr, val, nitems * sizeof(T), NULL,
            user_arg, &promise->internal);
    return promise->get_future();
}

template<typename T, typename... future_list_t>
future_t<void> *async_memset(place_t *pl, T *ptr, int val,
        size_t nitems, void *user_arg, future_list_t... futures) {
    hclib::promise_t<void> *promise = new hclib::promise_t<void>();
    hclib_future_t **future_list = construct_future_list(futures...);
    hclib_async_memset_helper(pl, ptr, val, nitems * sizeof(T), future_list,
            user_arg, &promise->internal);
    return promise->get_future();
}
#endif

#ifdef HUPCPP
int total_pending_local_asyncs();
volatile int *start_finish_special();
void end_finish(); // This is an ugly API, but must be exposed for HUPC
void display_runtime();
void get_avg_time(double* t_work, double *t_ovh, double* t_search);
void gather_comm_worker_stats(int* push_outd, int* push_ind,
        int* steal_ind);
#endif

#ifdef HCSHMEM
int total_pending_local_asyncs();
volatile int *start_finish_special();
#endif

}

#endif
