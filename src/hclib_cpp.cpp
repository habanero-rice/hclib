#include "hclib_cpp.h"

hclib::promise_t *hclib::promise_create() {
    return hclib_promise_create();
}

hclib::promise_t **hclib::promise_create_n(size_t nb_promises, int null_terminated) {
    return hclib_promise_create_n(nb_promises, null_terminated);
}

void hclib::promise_init(hclib::promise_t* promise) {
    hclib_promise_init(promise);
}

void hclib::promise_free(hclib::promise_t *promise) {
    hclib_promise_free(promise);
}

void hclib::promise_put(hclib::promise_t *promise, void *datum) {
    hclib_promise_put(promise, datum);
}

void *hclib::promise_get(hclib::promise_t *promise) {
    return hclib_promise_get(promise);
}

hc_workerState *hclib::current_ws() {
    return CURRENT_WS_INTERNAL;
}

int hclib::current_worker() {
    return get_current_worker();
}

int hclib::num_workers() {
    return hclib_num_workers();
}

int hclib::get_num_places(hclib::place_type_t type) {
    return hclib_get_num_places(type);
}

void hclib::get_places(hclib::place_t **pls, hclib::place_type_t type) {
    hclib_get_places(pls, type);
}

place_t *hclib::get_current_place() {
    return hclib_get_current_place();
}

place_t **hclib::get_children_of_place(place_t *pl, int *num_children) {
    return hclib_get_children_of_place(pl, num_children);
}

place_t *hclib::get_root_place() {
    return hclib_get_root_place();
}

place_t **hclib::get_nvgpu_places(int *n_nvgpu_places) {
    return hclib_get_nvgpu_places(n_nvgpu_places);
}

char *hclib::get_place_name(place_t *pl) {
    return hclib_get_place_name(pl);
}

void *hclib::promise_wait(hclib::promise_t *promise) {
    return hclib_promise_wait(promise);
}

#ifdef HUPCPP
int hclib::total_pending_local_asyncs() {
    return totalPendingLocalAsyncs();
}

volatile int *hclib::start_finish_special() {
    return hclib_start_finish_special();
}

void hclib::end_finish() {
    hclib_end_finish();
}

void hclib::display_runtime() {
    hclib_display_runtime();
}

void hclib::get_avg_time(double* tWork, double *tOvh, double* tSearch) {
    hclib_get_avg_time(tWork, tOvh, tSearch);
}

void hclib::gather_comm_worker_stats(int* push_outd, int* push_ind,
        int* steal_ind) {
    hclib_gather_comm_worker_stats(push_outd, push_ind, steal_ind);
}
#endif
