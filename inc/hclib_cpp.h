#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.h"
#include "hclib-asyncAwait.h"
#include "hclib-forasync.h"

namespace hclib {

typedef hclib_ddf_t ddf_t;
typedef hclib_ddt_t ddt_t;
typedef loop_domain_t loop_domain_t;
typedef place_t place_t;
typedef place_type_t place_type_t;

template <typename T>
void launch(int *argc, char **argv, T lambda) {
    hclib_task_t *user_task = _allocate_async(lambda, false);
    hclib_launch(argc, argv, (generic_framePtr)spawn, user_task);
}

ddf_t *ddf_create();
ddf_t **ddf_create_n(size_t nb_ddfs, int null_terminated);
void ddf_init(ddf_t* ddf);
void ddf_free(ddf_t *ddf);
void ddf_put(ddf_t *ddf, void *datum);
void *ddf_get(ddf_t *ddf);
void *ddf_wait(ddf_t *ddf);

hc_workerState *current_ws();
int current_worker();
int num_workers();
int get_num_places(place_type_t type);
void get_places(place_t **pls, place_type_t type);
place_t *get_current_place();
place_t **get_children_of_place(place_t *pl, int *num_children);
place_t *get_root_place();
place_t **get_nvgpu_places(int *n_nvgpu_places);
char *get_place_name(place_t *pl);

#ifdef HC_CUDA
template<typename T>
T *allocate_at(place_t *pl, size_t nitems, int flags) {
    return (T *)hclib_allocate_at(pl, nitems * sizeof(T), flags);
}

template<typename T>
void free_at(place_t *pl, T *ptr) {
    return hclib_free_at(pl, ptr);
}

template<typename T>
ddf_t *async_copy(place_t *dst_pl, T *dst,
        place_t *src_pl, T *src, size_t nitems,
        hclib_ddf_t **ddf_list, void *user_arg) {
    return hclib_async_copy(dst_pl, dst, src_pl, src, nitems * sizeof(T),
            ddf_list, user_arg);
}

template<typename T>
ddf_t *async_memset(place_t *pl, T *ptr, int val,
        size_t nitems, hclib_ddf_t **ddf_list, void *user_arg) {
    return hclib_async_memset(pl, ptr, val, nitems * sizeof(T), ddf_list,
            user_arg);
}
#endif

#ifdef HUPCPP
int total_pending_local_asyncs();
volatile int *start_finish_special();
void end_finish(); // This is an ugly API, but must be exposed for HUPC
void display_runtime();
void get_avg_time(double* tWork, double *tOvh, double* tSearch);
void gather_comm_worker_stats(int* push_outd, int* push_ind,
        int* steal_ind);
#endif

}

#endif
