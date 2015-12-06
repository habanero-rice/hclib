#ifndef HCPP_TASK_H_
#define HCPP_TASK_H_

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
 *   2) _args: a fixed-size buffer to hold the arguments to that function.
 *   3) current_finish: a pointer to the finish scope this task is registered on
 *      (possibly NULL).
 *   4) is_asyncAnyType: a boolean that doesn't seem to be ever be set to 1...
 *   5) ddf_list: a null-terminated list of pointers to the DDFs that this task
 *      depends on to execute, and which it will wait on before running.
 */
typedef struct _task_t {
    char _args[MAX_HCPP_ASYNC_ARG_SIZE];
    struct finish_t* current_finish;
    generic_framePtr _fp;
    /*
     * Boolean flag specific to HabaneroUPC++ only.
     * Used to differentiate between a normal hcpp async
     * and locality flexible asyncAny
     */
    int is_asyncAnyType;
    hclib_ddf_t ** ddf_list; // Null terminated list
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
