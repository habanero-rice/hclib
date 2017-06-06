#include "hclib-internal.h"
#include "hclib-module.h"

#include <assert.h>

extern hclib_context *hc_context;
static unsigned worker_state_size = 0;

typedef struct {
    void **elements;
    size_t length;
} simple_list;

static simple_list *simple_list_create() {
    simple_list *l = (simple_list *)malloc(sizeof(*l));
    assert(l);
    l->elements = NULL;
    l->length = 0;
    return l;
}

static int simple_list_contains(void *ele, simple_list *l) {
    for (int i = 0; i < l->length; i++) {
        if (l->elements[i] == ele) {
            return 1;
        }
    }
    return 0;
}

static void simple_list_append(void *ele, simple_list *l) {
    if (!simple_list_contains(ele, l)) {
        l->elements = (void **)realloc(l->elements,
                (l->length + 1) * sizeof(void *));
        assert(l->elements);
        (l->elements)[l->length] = ele;
        l->length += 1;
    }
}

static simple_list *pre_init_functions = NULL;
static simple_list *post_init_functions = NULL;
static simple_list *finalize_functions = NULL;

#ifdef __cplusplus
extern "C" {
#endif

int hclib_add_module_init_function(const char *name,
        hclib_module_pre_init_func_type pre,
        hclib_module_post_init_func_type post,
        hclib_module_finalize_func_type finalize) {
#ifdef VERBOSE
    printf("Registering module %s\n", name);
#endif

    if (pre_init_functions == NULL) {
        pre_init_functions = simple_list_create();
        post_init_functions = simple_list_create();
        finalize_functions = simple_list_create();
    }

    if (simple_list_contains(pre, pre_init_functions)) {
        assert(simple_list_contains(post, post_init_functions));
        assert(simple_list_contains(finalize, finalize_functions));
    } else {
        assert(!simple_list_contains(post, post_init_functions));
        assert(!simple_list_contains(finalize, finalize_functions));

        simple_list_append(pre, pre_init_functions);
        simple_list_append(post, post_init_functions);
        simple_list_append(finalize, finalize_functions);
    }

    return 0;
}

void hclib_call_module_pre_init_functions() {
    if (pre_init_functions == NULL) {
#ifdef VERBOSE
        printf("Pre-initializing 0 module(s)\n");
#endif
        return;
    }

#ifdef VERBOSE
    printf("Pre-initializing %d module(s)\n", pre_init_functions->length);
#endif
    for (int i = 0; i < pre_init_functions->length; i++) {
        hclib_module_pre_init_func_type curr = pre_init_functions->elements[i];
        (*curr)();
    }
}

void hclib_call_module_post_init_functions() {
    if (post_init_functions == NULL) {
#ifdef VERBOSE
        printf("Post-initializing 0 module(s)\n");
#endif
        return;
    }

#ifdef VERBOSE
    printf("Post-initializing %d module(s)\n", post_init_functions->length);
#endif
    for (int i = 0; i < post_init_functions->length; i++) {
        hclib_module_post_init_func_type curr = post_init_functions->elements[i];
        (*curr)();
    }
}

void hclib_call_finalize_functions() {
    if (finalize_functions == NULL) {
#ifdef VERBOSE
        printf("Finalizing 0 module(s)\n");
#endif
        return;
    }

#ifdef VERBOSE
    printf("Finalizing %d module(s)\n", finalize_functions->length);
#endif
    for (int i = 0; i < finalize_functions->length; i++) {
        hclib_module_finalize_func_type curr = finalize_functions->elements[i];
        (*curr)();
    }
}

unsigned hclib_add_per_worker_module_state(size_t state_size,
        hclib_state_adder cb, void *user_data) {
    int i;

    const unsigned offset = worker_state_size;

    for (i = 0; i < hc_context->nworkers; i++) {
        hclib_worker_state *ws = hc_context->workers[i];
        ws->module_state = (char *)realloc(ws->module_state,
                worker_state_size + state_size);
        assert(ws->module_state);

        cb(ws->module_state + offset, user_data, i);
    }

    worker_state_size += state_size;

    return offset;
}

void *hclib_get_curr_worker_module_state(const unsigned state_id) {
    hclib_worker_state *ws = current_ws();
    return ws->module_state + state_id;
}

// Normally called from a module's finalize function
void hclib_release_per_worker_module_state(const unsigned state_id,
        hclib_state_releaser cb, void *user_data) {
    int i;

    for (i = 0; i < hc_context->nworkers; i++) {
        hclib_worker_state *ws = hc_context->workers[i];
        char *state = ws->module_state + state_id;
        cb(state, user_data);
    }
}

#ifdef __cplusplus
}
#endif
