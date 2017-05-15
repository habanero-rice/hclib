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

#ifndef HCLIB_TASK_H_
#define HCLIB_TASK_H_

#include "hclib-rt.h"

/*
 * We just need to pack the function pointer and the pointer to
 * the heap allocated lambda for the lambda-based approach. Without lambdas, we
 * simply store the user-provided data pointer (which is obviously smaller).
 */
#define MAX_HCLIB_ASYNC_ARG_SIZE (sizeof(void *) + sizeof(void *))

/*
 * The core task representation, including:
 *
 *   1) _fp: a user-provided function pointer to execute.
 *   2) _args: a pointer to user-provided arguments to that function.
 *   3) current_finish: a pointer to the finish scope this task is registered on
 *      (possibly NULL).
 *   4) future_frontier: index of next awaited future in list
 *   5) future_list: a null-terminated list of pointers to the futures that
 *      this task depends on to execute, and which it will wait on before
 *      running.
 */
typedef struct hclib_task_t {
    void *args;
    struct finish_t *current_finish;
    generic_frame_ptr _fp;
    int future_frontier; // index of next awaited future in list
    hclib_future_t **future_list; // Null terminated list
    place_t *place;
    struct hclib_task_t *next_waiter;
} hclib_task_t;

/** @struct loop_domain_t
 * @brief Describe loop domain when spawning a forasync.
 * @param[in] low       Lower bound for the loop
 * @param[in] high      Upper bound for the loop
 * @param[in] stride    Stride access
 * @param[in] tile      Tile size for chunking
 */
typedef struct {
    int low;
    int high;
    int stride;
    int tile;
} loop_domain_t;

typedef struct {
    hclib_task_t *user;
} forasync_t;

typedef struct {
    forasync_t base;
    loop_domain_t loop0;
} forasync1D_t;

typedef struct _forasync_1D_task_t {
    hclib_task_t forasync_task;
    forasync1D_t def;
} forasync1D_task_t;

typedef struct {
    forasync_t base;
    loop_domain_t loop0;
    loop_domain_t loop1;
} forasync2D_t;

typedef struct _forasync_2D_task_t {
    hclib_task_t forasync_task;
    forasync2D_t def;
} forasync2D_task_t;

typedef struct {
    forasync_t base;
    loop_domain_t loop0;
    loop_domain_t loop1;
    loop_domain_t loop2;
} forasync3D_t;

typedef struct _forasync_3D_task_t {
    hclib_task_t forasync_task;
    forasync3D_t def;
} forasync3D_task_t;

#endif
