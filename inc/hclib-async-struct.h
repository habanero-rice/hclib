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

#ifndef HCLIB_ASYNCSTRUCT_H_
#define HCLIB_ASYNCSTRUCT_H_

#include <string.h>

#include "hclib-place.h"
#include "hclib-task.h"

#ifdef __cplusplus
extern "C" {
#endif

inline hclib_future_t ** get_future_list(hclib_task_t *t) {
    return t->future_list;
}

inline void mark_as_async_any_task(hclib_task_t *t) {
    t->is_async_any_type = 1;
}

inline int is_async_any_task(hclib_task_t *t) {
    return t->is_async_any_type;
}

void spawn(hclib_task_t * task);
void spawn_at_hpt(place_t* pl, hclib_task_t * task);
void spawn_await_at(hclib_task_t * task, hclib_future_t** future_list,
        place_t *pl);
void spawn_await(hclib_task_t * task, hclib_future_t** future_list);
void spawn_escaping(hclib_task_t * task, hclib_future_t** future_list);
void spawn_comm_task(hclib_task_t * task);
void spawn_gpu_task(hclib_task_t *task);

#ifdef __cplusplus
}
#endif

#endif /* HCLIB_ASYNCSTRUCT_H_ */
