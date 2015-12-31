/*
 * hclib-asyncStruct.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

#ifndef HCLIB_ASYNCSTRUCT_H_
#define HCLIB_ASYNCSTRUCT_H_

#include <string.h>

#include "hclib-place.h"
#include "hclib-task.h"

#ifdef __cplusplus
extern "C" {
#endif

inline hclib_promise_t ** get_promise_list(hclib_task_t *t) {
    return t->promise_list;
}

inline void mark_as_asyncAnyTask(hclib_task_t *t) {
    t->is_asyncAnyType = 1;
}

inline int is_asyncAnyTask(hclib_task_t *t) {
    return t->is_asyncAnyType;
}

void spawn(hclib_task_t * task);
void spawn_at_hpt(place_t* pl, hclib_task_t * task);
void spawn_await_at(hclib_task_t * task, hclib_promise_t** promise_list, place_t *pl);
void spawn_await(hclib_task_t * task, hclib_promise_t** promise_list);
void spawn_commTask(hclib_task_t * task);
void spawn_gpu_task(hclib_task_t *task);

#ifdef __cplusplus
}
#endif

#endif /* HCLIB_ASYNCSTRUCT_H_ */
