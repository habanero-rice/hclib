#include "hclib-internal.h"
#include "hclib-module.h"

#include <algorithm>
#include <assert.h>
#include <vector>
#include <iostream>

extern hclib_context *hc_context;
static unsigned worker_state_size = 0;

static std::vector<hclib_module_pre_init_func_type> *pre_init_functions =
        NULL;
static std::vector<hclib_module_pre_init_func_type> *post_init_functions =
        NULL;
static std::vector<hclib_module_finalize_func_type> *finalize_functions = NULL;

#ifdef __cplusplus
extern "C" {
#endif

int hclib_add_module_init_function(const char *name,
        hclib_module_pre_init_func_type pre,
        hclib_module_post_init_func_type post,
        hclib_module_finalize_func_type finalize) {
    std::cout << "Registering module " << name << std::endl;

    if (pre_init_functions == NULL) {
        pre_init_functions = new std::vector<hclib_module_pre_init_func_type>();
        post_init_functions =
            new std::vector<hclib_module_post_init_func_type>();
        finalize_functions = new std::vector<hclib_module_finalize_func_type>();
    }

    if (std::find(pre_init_functions->begin(), pre_init_functions->end(),
                pre) != pre_init_functions->end()) {
        assert(std::find(post_init_functions->begin(),
                    post_init_functions->end(), post) !=
                post_init_functions->end());
        assert(std::find(finalize_functions->begin(), finalize_functions->end(),
                    finalize) != finalize_functions->end());
    } else {
        assert(std::find(post_init_functions->begin(),
                    post_init_functions->end(), post) ==
                post_init_functions->end());
        assert(std::find(finalize_functions->begin(), finalize_functions->end(),
                    finalize) == finalize_functions->end());

        pre_init_functions->push_back(pre);
        post_init_functions->push_back(post);
        finalize_functions->push_back(finalize);
    }

    return 0;
}

void hclib_call_module_pre_init_functions() {
    if (pre_init_functions == NULL) {
        std::cout << "Pre-initializing 0 module(s)" << std::endl;
        return;
    }

    std::cout << "Pre-initializing " << pre_init_functions->size() <<
        " module(s)" << std::endl;
    for (std::vector<hclib_module_pre_init_func_type>::iterator i =
            pre_init_functions->begin(), e =
            pre_init_functions->end(); i != e; i++) {
        hclib_module_pre_init_func_type curr = *i;
        (*curr)();
    }
}

void hclib_call_module_post_init_functions() {
    if (post_init_functions == NULL) {
        std::cout << "Post-initializing 0 module(s)" << std::endl;
        return;
    }

    std::cout << "Post-initializing " << post_init_functions->size() <<
        " module(s)" << std::endl;
    for (std::vector<hclib_module_post_init_func_type>::iterator i =
            post_init_functions->begin(), e =
            post_init_functions->end(); i != e; i++) {
        hclib_module_post_init_func_type curr = *i;
        (*curr)();
    }
}

void hclib_call_finalize_functions() {
    if (finalize_functions == NULL) {
        std::cout << "Finalizing 0 module(s)" << std::endl;
        return;
    }

    std::cout << "Finalizing " << finalize_functions->size() << " module(s)" <<
        std::endl;
    for (std::vector<hclib_module_finalize_func_type>::iterator i =
            finalize_functions->begin(), e = finalize_functions->end(); i != e;
            i++) {
        hclib_module_finalize_func_type curr = *i;
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

        cb(ws->module_state + offset, user_data);
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
