#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.h"
#include "hclib-forasync.h"
#include "hclib_promise.h"
#include "hclib.h"

namespace hclib {

typedef hclib_triggered_task_t triggered_task_t;
typedef loop_domain_t loop_domain_t;
typedef place_t place_t;
typedef place_type_t place_type_t;

template <typename T>
void launch(int *argc, char **argv, T lambda) {
    hclib_task_t *user_task = _allocate_async(lambda, false);
    hclib_launch(argc, argv, (generic_frame_ptr)spawn, user_task);
}

promise_t **promise_create_n(size_t nb_promises, int null_terminated);

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
future_t *async_copy(place_t *dst_pl, T *dst, place_t *src_pl, T *src,
        size_t nitems, void *user_arg) {
    hclib::promise_t *promise = new hclib::promise_t();
    hclib_async_copy_helper(dst_pl, dst, src_pl, src, nitems * sizeof(T),
            NULL, user_arg, &promise->internal);
    return promise->get_future();
}

template<typename T, typename... future_list_t>
future_t *async_copy(place_t *dst_pl, T *dst,
        place_t *src_pl, T *src, size_t nitems,
        void *user_arg, future_list_t... futures) {
    hclib::promise_t *promise = new hclib::promise_t();
    hclib_future_t **future_list = construct_future_list(futures...);
    hclib_async_copy_helper(dst_pl, dst, src_pl, src, nitems * sizeof(T),
            future_list, user_arg, &promise->internal);
    return promise->get_future();
}

template<typename T>
future_t *async_memset(place_t *pl, T *ptr, int val,
        size_t nitems, void *user_arg) {
    hclib::promise_t *promise = new hclib::promise_t();
    hclib_async_memset_helper(pl, ptr, val, nitems * sizeof(T), NULL,
            user_arg, &promise->internal);
    return promise->get_future();
}

template<typename T, typename... future_list_t>
future_t *async_memset(place_t *pl, T *ptr, int val,
        size_t nitems, void *user_arg, future_list_t... futures) {
    hclib::promise_t *promise = new hclib::promise_t();
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
