#ifndef HCLIB_TASK_H_
#define HCLIB_TASK_H_

#include "hclib-rt.h"
#include "hclib-locality-graph.h"

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
 *   2) args: a pointer to user-provided arguments to that function.
 *   3) current_finish: a pointer to the finish scope this task is registered on
 *      (possibly NULL).
 *   4) waiting_on: a list of futures on which this task is dependent, up to a
 *      maximum of MAX_NUM_WAITS. This array is initialized to NULL, and only
 *      non-NULL entries are valid. This array is always terminated by a NULL
 *      entry.
 *   5) waiting_on_index: The future in waiting_on that we are currently waiting
 *      for.
 *   5) locale: The locale at which this task should execute.
 *   6) non_blocking: Whether this task will block on other operations (i.e.
 *      call hclib_end_finish, hclib_future_wait, etc).
 *   7) next_waiter: Used to track tasks blocked on the same future.
 */
typedef struct hclib_task_t {
    generic_frame_ptr _fp;
    void *args;
    struct finish_t *current_finish;
    hclib_future_t *waiting_on[MAX_NUM_WAITS + 1];
    int waiting_on_index;
    hclib_locale_t *locale;
    int non_blocking;
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
} hclib_loop_domain_t;

/*
 * A function which accepts:
 *
 *   1) The dimensionality of a loop (1, 2, or 3).
 *   2) The domain of each dimension of a subset of that loop that is being
 *      scheduled.
 *   3) The domain of each dimension of the whole loop.
 *   4) The loop execution mode (recursive or flat)
 *   5) Some arbitrary, user-provided data.
 *
 * and which then returns a locale to place this subset of the loop at.
 */
typedef hclib_locale_t *(*loop_dist_func)(const int,
        const hclib_loop_domain_t *, const hclib_loop_domain_t *, const int);

typedef struct {
    hclib_task_t *user;
} forasync_t;

typedef struct {
    forasync_t base;
    hclib_loop_domain_t loop;
} forasync1D_t;

typedef struct _forasync_1D_task_t {
    hclib_task_t forasync_task;
    forasync1D_t def;
} forasync1D_task_t;

typedef struct {
    forasync_t base;
    hclib_loop_domain_t loop[2];
} forasync2D_t;

typedef struct _forasync_2D_task_t {
    hclib_task_t forasync_task;
    forasync2D_t def;
} forasync2D_task_t;

typedef struct {
    forasync_t base;
    hclib_loop_domain_t loop[3];
} forasync3D_t;

typedef struct _forasync_3D_task_t {
    hclib_task_t forasync_task;
    forasync3D_t def;
} forasync3D_task_t;

static inline struct finish_t* get_current_finish(hclib_task_t *t) {
    return t->current_finish;
}

static inline void set_current_finish(hclib_task_t *t,
        struct finish_t* finish) {
    t->current_finish = finish;
}

#endif
