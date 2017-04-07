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

#include "hclib_cpp.h"

hclib_worker_state *hclib::current_ws() {
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

#ifdef HC_CUDA
place_t **hclib::get_nvgpu_places(int *n_nvgpu_places) {
    return hclib_get_nvgpu_places(n_nvgpu_places);
}
#endif

char *hclib::get_place_name(place_t *pl) {
    return hclib_get_place_name(pl);
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

void hclib::get_avg_time(double *tWork, double *tOvh, double *tSearch) {
    hclib_get_avg_time(tWork, tOvh, tSearch);
}

void hclib::gather_comm_worker_stats(int *push_outd, int *push_ind,
                                     int *steal_ind) {
    hclib_gather_comm_worker_stats(push_outd, push_ind, steal_ind);
}
#endif

#ifdef HCSHMEM

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

void hclib::gather_comm_worker_stats(int *push_outd, int *push_ind,
                                     int *steal_ind) {
    hclib_gather_comm_worker_stats(push_outd, push_ind, steal_ind);
}

#endif

