#ifndef HCPP_TASK_H_
#define HCPP_TASK_H_

/*
 * We just need to pack the function pointer and the pointer to
 * the heap allocated lambda. Due to this 16 bytes is sufficient
 * for our lambda based approach.
 */
#define MAX_HCPP_ASYNC_ARG_SIZE 16

struct hcpp_async_task  {
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
};

#ifdef __cplusplus
typedef struct hcpp_async_task task_t;
#else
#define task_t         struct hcpp_async_task
#endif

typedef struct hcpp_task_t {
	task_t async_task; 	// the actual task
	ddt_t ddt; 			// ddt meta-information
} hcpp_task_t;

inline struct finish_t* get_current_finish(struct hcpp_async_task *t) {
    return t->current_finish;
}

inline void set_current_finish(struct hcpp_async_task *t,
        struct finish_t* finish) {
    t->current_finish = finish;
}

inline void set_ddf_list(struct hcpp_async_task *t, hclib_ddf_t ** ddf) {
    t->ddf_list = ddf;
}

#endif
