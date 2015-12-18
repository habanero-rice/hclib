#ifndef HCPP_TASK_H_
#define HCPP_TASK_H_

#include "hcpp-rt.h"

/*
 * We just need to pack the function pointer and the pointer to
 * the heap allocated lambda for the lambda-based approach. Without lambdas, we
 * simply store the user-provided data pointer (which is obviously smaller).
 */
#define MAX_HCPP_ASYNC_ARG_SIZE (sizeof(void *) + sizeof(void *))

/*
 * The core task representation, including:
 *
 *   1) _fp: a user-provided function pointer to execute.
 *   2) _args: a pointer to user-provided arguments to that function.
 *   3) current_finish: a pointer to the finish scope this task is registered on
 *      (possibly NULL).
 *   4) is_asyncAnyType: a boolean that doesn't seem to be ever be set to 1...
 *   5) ddf_list: a null-terminated list of pointers to the DDFs that this task
 *      depends on to execute, and which it will wait on before running.
 */
typedef struct _task_t {
    void *args;
    struct finish_t *current_finish;
    generic_framePtr _fp;
    /*
     * Boolean flag specific to HabaneroUPC++ only.
     * Used to differentiate between a normal hcpp async
     * and locality flexible asyncAny
     */
    int is_asyncAnyType;
    hclib_ddf_t **ddf_list; // Null terminated list
    place_t *place;
} task_t;

/*
 * A representation of a task whose execution is dependent on prior tasks
 * through a list of DDF objects.
 */
typedef struct hcpp_task_t {
	task_t async_task; 	// the actual task
    /*
     * ddt meta-information, tasks that this task is blocked on for execution.
     * ddt.waitingFrontier is generally equal to async_task.ddf_list. TODO can
     * we factor out this redundant storage of data?
     */
	ddt_t ddt;
} hcpp_task_t;

/** @struct loop_domain_t
 * @brief Describe loop domain when spawning a forasync.
 * @param[in] low       Lower bound for the loop
 * @param[in] high      Upper bound for the loop
 * @param[in] stride    Stride access
 * @param[in] tile      Tile size for chunking
 */
typedef struct _loop_domain_t {
    int low;
    int high;
    int stride;
    int tile;
} loop_domain_t;

typedef struct {
    task_t *user;
} forasync_t;

typedef struct {
    forasync_t base;
    loop_domain_t loop0;
} forasync1D_t;

typedef struct _forasync_1D_task_t {
    task_t forasync_task;
    forasync1D_t def;
} forasync1D_task_t;

typedef struct {
    forasync_t base;
    loop_domain_t loop0;
    loop_domain_t loop1;
} forasync2D_t;

typedef struct _forasync_2D_task_t {
    task_t forasync_task;
    forasync2D_t def;
} forasync2D_task_t;

typedef struct {
    forasync_t base;
    loop_domain_t loop0;
    loop_domain_t loop1;
    loop_domain_t loop2;
} forasync3D_t;

typedef struct _forasync_3D_task_t {
    task_t forasync_task;
    forasync3D_t def;
} forasync3D_task_t;

inline struct finish_t* get_current_finish(task_t *t) {
    return t->current_finish;
}

inline void set_current_finish(task_t *t,
        struct finish_t* finish) {
    t->current_finish = finish;
}

inline void set_ddf_list(task_t *t, hclib_ddf_t ** ddf) {
    t->ddf_list = ddf;
}

#endif
