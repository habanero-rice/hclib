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

#ifndef HCLIB_HPT_H_
#define HCLIB_HPT_H_

#include "hclib-internal.h"

place_t * read_hpt(place_t *** all_places, int * num_pl, int * nproc,
        hclib_worker_state *** all_workers, int * num_wk);
void free_hpt(place_t * hpt);
void hc_hpt_init(hc_context * context);
void hc_hpt_cleanup(hc_context * context);
void hc_hpt_dev_init(hc_context * context);
void hc_hpt_dev_cleanup(hc_context * context);
hc_deque_t * get_deque_place(hclib_worker_state * ws, place_t * pl);
hclib_task_t* hpt_pop_task(hclib_worker_state * ws);
hclib_task_t* hpt_steal_task(hclib_worker_state* ws);
int deque_push_place(hclib_worker_state *ws, place_t * pl, hclib_task_t * ele);

#endif /* HCLIB_HPT_H_ */
