/* Copyright (c) 2015, Rice University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
3.  Neither the name of Rice University
     nor the names of its contributors may be used to endorse or
     promote products derived from this software without specific
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

/*
 * hclib-hpt.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include <stdio.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <unistd.h>

#include "hclib-hpt.h"
#include "hclib-internal.h"
#include "hclib-atomics.h"
#include "hcupc-support.h"
#include "hclib-cuda.h"

// #define VERBOSE

inline hc_deque_t *get_deque_place(hclib_worker_state *ws, place_t *pl);
void free_hpt(place_t *hpt);
static const char *place_type_to_str(short type);

extern hc_context *hclib_context;

void *unsupported_place_type_err(place_t *pl) {
    fprintf(stderr, "Unsupported place type %s\n", place_type_to_str(pl->type));
    exit(1);
}

/**
 * HPT: Try to steal a frame from another worker.
 * 1) First look for work in current place worker deques
 * 2) If unsuccessful, start over at step 1) in the parent
 *    place all to the hpt top.
 */
hclib_task_t *hpt_steal_task(hclib_worker_state *ws) {
    MARK_SEARCH(ws->id); // Set the state of this worker for timing

    hcupc_reset_asyncAnyInfo(ws->id);

    place_t *pl = ws->pl;
    while (pl != NULL) {
        hc_deque_t *deqs = pl->deques;
        int nb_deq = pl->ndeques;

        /* Try to steal once from every other worker first */
        for (int i=1; i<nb_deq; i++) {
            int victim = ((ws->id + i) % nb_deq);
            hc_deque_t *d = &deqs[victim];
            hclib_task_t *buff = deque_steal(&(d->deque));
            if (buff) { /* steal succeeded */
                ws->current = get_deque_place(ws, pl);
                hcupc_check_if_asyncAny_stolen(buff, victim, ws->id);

#ifdef VERBOSE
                printf("hpt_steal_task: worker %d successful steal from deque %p, pl %p, "
                       "level %d\n", ws->id, d, d->pl, d->pl->level);
#endif
                return buff;
            }
        }
        hcupc_inform_failedSteal(ws->id);

        /* Nothing found in this place, go to the parent */
        pl = pl->parent;
    }
    return NULL;
}

/**
 * HPT: Pop items from a worker deque
 * 1) Try to pop from current queue (Q)
 * 2) if nothing found, try to pop downward from worker's child deque.
 * 3) If nothing found down to the bottom, look upward starting from Q.
 */
hclib_task_t *hpt_pop_task(hclib_worker_state *ws) {
    // go HPT downward and then upward of my own deques
    hc_deque_t *current = ws->current;
    hc_deque_t *pivot = current;
    short downward = 1;

    while (current != NULL) {
        hclib_task_t *buff = deque_pop(&current->deque);
        if (buff) {
#ifdef VERBOSE
            printf("hpt_pop_task: worker %d successful pop from deque %p, pl %p, level "
                   "%d\n", ws->id ,current, current->pl, current->pl->level);
#endif
            ws->current = current;
            hcupc_check_if_asyncAny_pop(buff, ws->id);
            return buff;
        }
        if (downward) {
            if (current->nnext == NULL) {
                current = pivot->prev;
                downward = 0; // next, we go upward from pivot
            } else {
                current = current->nnext;
            }
        } else {
            current = current->prev;
        }
    }
    return NULL;
}

place_t *hclib_get_current_place() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    HASSERT(ws->current->pl != NULL);
    return ws->current->pl;
}

#ifdef HC_CUDA
place_t **hclib_get_nvgpu_places(int *n_nvgpu_places) {
    place_t *root_pl = hclib_get_root_place();

    int num_toplevel;
    place_t **toplevel = hclib_get_children_of_place(root_pl, &num_toplevel);

    int num_gpus = 0;
    int i;
    for (i = 0; i < num_toplevel; i++) {
        if (toplevel[i]->type == NVGPU_PLACE) {
            num_gpus++;
        }
    }

    num_gpus = 0;
    place_t **gpu_places = malloc(num_gpus * sizeof(place_t *));
    for (i = 0; i < num_toplevel; i++) {
        if (toplevel[i]->type == NVGPU_PLACE) {
            gpu_places[num_gpus++] = toplevel[i];
        }
    }

    *n_nvgpu_places = num_gpus;
    return gpu_places;
}
#endif

const char *cpu_place_name = "CPU";
char *hclib_get_place_name(place_t *pl) {
    if (is_cpu_place(pl)) {
        return (char *)cpu_place_name;
#ifdef HC_CUDA
    } else if (is_nvgpu_place(pl)) {
        struct cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, pl->cuda_id));
        char *gpu_name = (char *)malloc(sizeof(props.name));
        memcpy(gpu_name, props.name, sizeof(props.name));
        return gpu_name;
#endif
    } else {
        return unsupported_place_type_err(pl);
    }
}

int hclib_get_num_places(short type) {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t **all_places = ws->context->places;
    int np = ws->context->nplaces;
    int i;
    int num = 0;
    for (i=0; i<np; i++)
        if (all_places[i]->type == type) num++;
    return num;
}

void hclib_get_places(place_t **pls, short type) {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t **all_places = ws->context->places;
    int np = ws->context->nplaces;
    int i;
    int num = 0;
    for (i=0; i<np; i++)
        if (all_places[i]->type == type) pls[num++] = all_places[i];
    return;
}

place_t *hc_get_place(short type) {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t **all_places = ws->context->places;
    int np = ws->context->nplaces;
    int i;
    for (i=0; i<np; i++)
        if (all_places[i]->type == type)  return all_places[i];
    return NULL;
}

place_t *hclib_get_root_place() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t **all_places = ws->context->places;
    return all_places[0];
}

inline place_t *get_ancestor_place(hclib_worker_state *ws) {
    place_t *parent = ws->pl;
    while (parent->parent != NULL) parent = parent->parent;
    return parent;
}

place_t *hclib_get_child_place() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t *pl = ws->current->pl;
    HASSERT(pl != NULL);
    if (ws->hpt_path == NULL) return pl;
    return ws->hpt_path[pl->level + 1];
}

place_t *hclib_get_parent_place() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t *pl = ws->current->pl;
    HASSERT(pl != NULL);
    if (ws->hpt_path == NULL) return pl;
    return ws->hpt_path[pl->level - 1];
}

place_t **hclib_get_children_places(int *numChildren) {
    place_t *pl = hclib_get_current_place();
    *numChildren = pl->nChildren;
    return pl->children;
}

place_t **hclib_get_children_of_place(place_t *pl, int *numChildren) {
    *numChildren = pl->nChildren;
    return pl->children;
}

/*
 * Return my own deque that is in place pl,
 * if pl == NULL, return the current deque of the worker
 */
inline hc_deque_t *get_deque_place(hclib_worker_state *ws, place_t *pl) {
    if (pl == NULL) return ws->current;
    return &(pl->deques[ws->id]);
}

hc_deque_t *get_deque(hclib_worker_state *ws) {
    return NULL;
}

/* get the first owned deque from HPT that starts with place pl upward */
hc_deque_t *get_deque_hpt(hclib_worker_state *ws, place_t *pl) {
    return NULL;

}

int deque_push_place(hclib_worker_state *ws, place_t *pl, void *ele) {
    hc_deque_t *deq = get_deque_place(ws, pl);
    return deque_push(&deq->deque, ele);
}

inline hclib_task_t *deque_pop_place(hclib_worker_state *ws, place_t *pl) {
    hc_deque_t *deq = get_deque_place(ws, pl);
    return deque_pop(&deq->deque);
}

/**
 * Initializes a hc_deque_t
 */
inline void init_hc_deque_t(hc_deque_t *hcdeq, place_t *pl) {
    hcdeq->deque.head = hcdeq->deque.tail = 0;
    hcdeq->pl = pl;
    hcdeq->ws = NULL;
    hcdeq->nnext = NULL;
    hcdeq->prev = NULL;
#ifdef BUCKET_DEQUE
    hcdeq->deque.last = 0;
    hcdeq->deque.thief = 0;
    hcdeq->deque.staleMaps = NULL;
#endif
}

#ifdef HC_CUDA
void *hclib_allocate_at(place_t *pl, size_t nbytes, int flags) {
    HASSERT(pl);
    HASSERT(nbytes > 0);
#ifdef VERBOSE
    fprintf(stderr, "hclib_allocate_at: pl=%p nbytes=%lu flags=%d, is_cpu? %s",
            pl, (unsigned long)nbytes, flags,
            is_cpu_place(pl) ? "true" : "false");
#ifdef HC_CUDA
    fprintf(stderr, ", is_nvgpu? %s, cuda_id=%d",
            is_nvgpu_place(pl) ? "true" : "false", pl->cuda_id);
#endif
    fprintf(stderr, "\n");
#endif

    if (is_cpu_place(pl)) {
#ifdef HC_CUDA
        if (flags & PHYSICAL) {
            void *ptr;
            const cudaError_t alloc_err = cudaMallocHost((void **)&ptr, nbytes);
            if (alloc_err != cudaSuccess) {
#ifdef VERBOSE
                fprintf(stderr, "Physical allocation at CPU place failed with "
                        "reason \"%s\"\n", cudaGetErrorString(alloc_err));
#endif
                return NULL;
            } else {
                hclib_memory_tree_insert(ptr, nbytes,
                                         &hclib_context->pinned_host_allocs);
                return ptr;
            }
        }
#else
        HASSERT(flags == NONE);
#endif
        return malloc(nbytes);
#ifdef HC_CUDA
    } else if (is_nvgpu_place(pl)) {
        HASSERT(flags == NONE);
        void *ptr;
        HASSERT(pl->cuda_id >= 0);
        CHECK_CUDA(cudaSetDevice(pl->cuda_id));
        const cudaError_t alloc_err = cudaMalloc((void **)&ptr, nbytes);
        if (alloc_err != cudaSuccess) {
#ifdef VERBOSE
            fprintf(stderr, "Allocation at NVGPU place failed with reason "
                    "\"%s\"\n", cudaGetErrorString(alloc_err));
#endif
            return NULL;
        } else {
            return ptr;
        }
#endif
    } else {
        unsupported_place_type_err(pl);
        return NULL; // will never reach here
    }
}

int is_pinned_cpu_mem(void *ptr) {
    return hclib_memory_tree_contains(ptr, &hclib_context->pinned_host_allocs);
}

void hclib_free_at(place_t *pl, void *ptr) {
    if (is_cpu_place(pl)) {
        if (is_pinned_cpu_mem(ptr)) {
            hclib_memory_tree_remove(ptr, &hclib_context->pinned_host_allocs);
            CHECK_CUDA(cudaFreeHost(ptr));
        } else {
            free(ptr);
        }
#ifdef HC_CUDA
    } else if (is_nvgpu_place(pl)) {
        CHECK_CUDA(cudaFree(ptr));
#endif
    } else {
        unsupported_place_type_err(pl);
    }
}

// Used to wrap the creation of an async copy task that is dependent on some promises
static void async_gpu_task_launcher(void *arg) {
    gpu_task_t *task = (gpu_task_t *)arg;
    spawn_gpu_task((hclib_task_t *)task);
}

/*
 * TODO Currently doesn't support await on other promises, nor do communication
 * tasks
 */

void hclib_async_copy_helper(place_t *dst_pl, void *dst, place_t *src_pl,
                             void *src, size_t nbytes, hclib_future_t **future_list,
                             void *user_arg, hclib_promise_t *out_promise) {
    gpu_task_t *task = malloc(sizeof(gpu_task_t));
    task->t._fp = NULL;
    task->t.is_asyncAnyType = 0;
    task->t.future_list = NULL;
    task->t.args = NULL;
    task->t.place = NULL;

    hclib_promise_init(out_promise);
    task->gpu_type = GPU_COMM_TASK;
    task->promise_to_put = out_promise;
    task->arg_to_put = user_arg;

    task->gpu_task_def.comm_task.src_pl = src_pl;
    task->gpu_task_def.comm_task.dst_pl = dst_pl;
    task->gpu_task_def.comm_task.src = src;
    task->gpu_task_def.comm_task.dst = dst;
    task->gpu_task_def.comm_task.nbytes = nbytes;

#ifdef VERBOSE
    fprintf(stderr, "hclib_async_copy: dst_pl=%p dst=%p src_pl=%p src=%p "
            "nbytes=%lu future_list=%p\n", dst_pl, dst, src_pl, src,
            (unsigned long)nbytes, future_list);
#endif

    if (future_list) {
        hclib_async(async_gpu_task_launcher, task, future_list, NULL, NULL, 0);
    } else {
        spawn_gpu_task((hclib_task_t *)task);
    }
}

hclib_promise_t *hclib_async_copy(place_t *dst_pl, void *dst, place_t *src_pl,
                                  void *src, size_t nbytes, hclib_future_t **future_list,
                                  void *user_arg) {
    hclib_promise_t *promise = hclib_promise_create();
    hclib_async_copy_helper(dst_pl, dst, src_pl, src, nbytes, future_list,
                            user_arg, promise);
    return promise;
}

void hclib_async_memset_helper(place_t *pl, void *ptr, int val, size_t nbytes,
                               hclib_future_t **future_list, void *user_arg,
                               hclib_promise_t *out_promise) {
    gpu_task_t *task = malloc(sizeof(gpu_task_t));
    task->t._fp = NULL;
    task->t.is_asyncAnyType = 0;
    task->t.future_list = NULL;
    task->t.args = NULL;
    task->t.place = NULL;

    hclib_promise_init(out_promise);
    task->gpu_type = GPU_MEMSET_TASK;
    task->promise_to_put = out_promise;
    task->arg_to_put = user_arg;

    task->gpu_task_def.memset_task.pl = pl;
    task->gpu_task_def.memset_task.ptr = ptr;
    task->gpu_task_def.memset_task.val = val;
    task->gpu_task_def.memset_task.nbytes = nbytes;

#ifdef VERBOSE
    fprintf(stderr, "hclib_async_memset: pl=%p ptr=%p nbytes=%lu\n", pl, ptr,
            (unsigned long)nbytes);
#endif

    if (future_list) {
        hclib_async(async_gpu_task_launcher, task, future_list, NULL, NULL, 0);
    } else {
        spawn_gpu_task((hclib_task_t *)task);
    }
}

hclib_promise_t *hclib_async_memset(place_t *pl, void *ptr, int val,
                                    size_t nbytes, hclib_future_t **future_list, void *user_arg) {
    hclib_promise_t *promise = hclib_promise_create();
    hclib_async_memset_helper(pl, ptr, val, nbytes, future_list, user_arg,
                              promise);
    return promise;
}
#endif

static const char *MEM_PLACE_STR   = "MEM_PLACE";
static const char *CACHE_PLACE_STR = "CACHE_PLACE";
static const char *NVGPU_PLACE_STR = "NVGPU_PLACE";
static const char *AMGPU_PLACE_STR = "AMGPU_PLACE";
static const char *FPGA_PLACE_STR  = "FPGA_PLACE";
static const char *PGAS_PLACE_STR  = "PGAS_PLACE";

static const char *place_type_to_str(short type) {
    switch (type) {
    case (MEM_PLACE):
        return MEM_PLACE_STR;
    case (CACHE_PLACE):
        return CACHE_PLACE_STR;
    case (NVGPU_PLACE):
        return NVGPU_PLACE_STR;
    case (AMGPU_PLACE):
        return AMGPU_PLACE_STR;
    case (FPGA_PLACE):
        return FPGA_PLACE_STR;
    case (PGAS_PLACE):
        return PGAS_PLACE_STR;
    default:
        fprintf(stderr, "unknown place type %d\n", type);
        exit(5);
    }
}

/* init the hpt and place deques */
void hc_hpt_init(hc_context *context) {
    int i, j;
#ifdef HPT_DESCENTWORKER_PERPLACE
    /*
     * each place has a deque for all workers beneath it (transitively) in the
     * HPT.
     */
    for (i = 0; i < context->nplaces; i++) {
        place_t *pl = context->places[i];
        int nworkers = pl->ndeques;
        pl->deques = malloc(sizeof(hc_deque_t) * nworkers);
        HASSERT(pl->deques);
        for (j = 0; j < nworkers; j++) {
            hc_deque_t *deq = &(pl->deques[j]);
            init_hc_deque_t(deq, pl);
        }
    }
#else // HPT_ALLWORKER_PERPLACE each place has a deque for each worker
    for (i = 0; i < context->nplaces; i++) {
        place_t *pl = context->places[i];
        const int ndeques = context->nworkers;
#ifdef TODO
        if (is_device_place(pl)) ndeques = 1;
#endif
        pl->ndeques = ndeques;
        pl->deques = (hc_deque_t *) malloc(sizeof(hc_deque_t) * ndeques);
        for (j = 0; j < ndeques; j++) {
            hc_deque_t *hc_deq = &(pl->deques[j]);
            init_hc_deque_t(hc_deq, pl);
        }
    }
#endif

    /*
     * link the deques for each cpu workers. the deque index is the same as
     * ws->id to simplify the search. For every worker, iterate over all places
     * and store a pointer from the place's deque for that worker to the worker
     * state for that worker.
     *
     * This builds a tree of deques from the worker, to its parent's deque for
     * it, to its grandparent's deque for it, up to the root. It would seem that
     * the majority of deques are therefore unused (i.e. even though we allocate
     * a dequeue for every worker in a platform in every place, only the deques
     * for workers that are beneath that place in the HPT are used). However,
     * this does make lookups of the deque in a place for a given worker
     * constant time based on offset in place->deques.
     */
#ifdef HC_CUDA
    int ngpus = 0;
    int gpu_counter = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&ngpus);
    if (cuda_err == cudaErrorNoDevice) {
        ngpus = 0;
    }

    for (i = 0; i < context->nplaces; i++) {
        place_t *pl = context->places[i];
        pl->cuda_id = -1;
        if (is_nvgpu_place(pl)) {
            pl->cuda_id = gpu_counter++;
            CHECK_CUDA(cudaSetDevice(pl->cuda_id));
            CHECK_CUDA(cudaStreamCreate(&pl->cuda_stream));
        }
    }
#endif

    for (i = 0; i < context->nworkers; i++) {
        hclib_worker_state *ws = context->workers[i];
        const int id = ws->id;
        for (j = 0; j < context->nplaces; j++) {
            place_t *pl = context->places[j];
            if (is_cpu_place(pl)) {
                hc_deque_t *hc_deq = &(pl->deques[id]);
                hc_deq->ws = ws;
#ifdef HC_CUDA
            } else if (is_nvgpu_place(pl)) {
                hc_deque_t *hc_deq = &(pl->deques[id]);
                hc_deq->ws = ws;

#endif
            } else {
                /* unhandled or ignored situation */
                HASSERT(0);
            }
        }

        /* here we link the deques of the ancestor places for this worker */
        place_t *parent = ws->pl;
        place_t *current = parent;
        ws->deques = &(current->deques[id]);
        while (parent->parent != NULL) {
            parent = parent->parent;
            current->deques[id].prev = &(parent->deques[id]);
            parent->deques[id].nnext = &(current->deques[id]);
            current = parent;
        }
        ws->current = &(current->deques[id]);
    }

#ifdef VERBOSE
    /*Print HPT*/
    int level = context->places[0]->level;
    printf("Level %d: ", level);
    for (i = 0; i < context->nplaces; i++) {
        place_t *pl = context->places[i];
        if (level != pl->level) {
            printf("\n");
            level = pl->level;
            printf("Level %d: ", level);
        }

        printf("Place %d %s ", pl->id, place_type_to_str(pl->type));
        hclib_worker_state *w = pl->workers;
        if (w != NULL) {
            printf("[ ");
            while (w != NULL) {
                printf("%d ", w->id);
                w = w->next_worker;
            }
            printf("] ");
        }

        place_t *c = pl->child;
        if (c != NULL) {
            printf("{ ");
            while (c != NULL) {
                printf("%d ", c->id);
                c = c->nnext;
            }
            printf("} ");
        }
        printf("\t");
    }
    printf("\n");
#endif
}

void hc_hpt_cleanup(hc_context *context) {
    /* clean up HPT deques for cpu workers*/
    for (int i = 0; i < context->nplaces; i++) {
        place_t *pl = context->places[i];
#ifdef TODO
        if (is_device_place(pl)) continue;
#endif
        free(pl->deques);
    }
    /* clean up the HPT, places and workers */
    free_hpt(context->hpt);
}

/*
 * Interfaces to read the xml files and parse correctly to generate the place data-structures
 */
hclib_worker_state *parse_worker_element(xmlNode *wkNode) {
    hclib_worker_state *wk = (hclib_worker_state *) malloc(sizeof(
                                 hclib_worker_state));
    memset(wk, 0x00, sizeof(hclib_worker_state));

    xmlChar *num = xmlGetProp(wkNode, xmlCharStrdup("num"));
    xmlChar *didStr = xmlGetProp(wkNode, xmlCharStrdup("did"));
    xmlChar *type = xmlGetProp(wkNode, xmlCharStrdup("type"));

    /*
     * Kind of hacky, the id field is re-used to store the number of workers at
     * this level in the hierarchy. This field is later re-assigned with the
     * actual worker ID.
     */
    if (num != NULL) wk->id = atoi((char *)num);
    else wk->id = 1;

    if (didStr != NULL) wk->did = atoi((char *)didStr);
    else wk->did = 0;
    wk->next_worker = NULL;
    /* TODO: worker/deque type */

    xmlFree(num);
    xmlFree(didStr);
    xmlFree(type);
    return wk;
}

place_t *parse_place_element(xmlNode *plNode) {
    int num = 1;
    int did = 0;
    xmlChar *numStr = xmlGetProp(plNode, xmlCharStrdup("num"));
    xmlChar *didStr = xmlGetProp(plNode, xmlCharStrdup("did"));
    xmlChar *typeStr = xmlGetProp(plNode, xmlCharStrdup("type"));
    xmlChar *sizeStr = xmlGetProp(plNode, xmlCharStrdup("size"));
    xmlChar *unitSize = xmlGetProp(plNode, xmlCharStrdup("unitSize"));
    xmlChar *info = xmlGetProp(plNode, xmlCharStrdup("info"));

    if (numStr != NULL) {
        num = atoi((char *)numStr);
    }

    if (didStr != NULL) {
        did = atoi((char *)didStr);
    }

    short type = CACHE_PLACE;
    if (typeStr != NULL) {
        if (!xmlStrcmp(typeStr, (const xmlChar *) "mem")) {
            type = MEM_PLACE;
        } else if (!xmlStrcmp(typeStr, (const xmlChar *) "cache")) {
            type = CACHE_PLACE;
        } else if (!xmlStrcmp(typeStr, (const xmlChar *) "nvgpu")) {
            type = NVGPU_PLACE;
        } else if (!xmlStrcmp(typeStr, (const xmlChar *) "amgpu")) {
            type = AMGPU_PLACE;
        } else if (!xmlStrcmp(typeStr, (const xmlChar *) "fpga")) {
            type = FPGA_PLACE;
        } else if (!xmlStrcmp(typeStr, (const xmlChar *) "pgas")) {
            type = PGAS_PLACE;
        } else {
            /* warning, unknown type specified */
        }
    } else {
        /* default to be cache type */
        type = CACHE_PLACE;
    }

    // Not supported yet
    HASSERT(type != AMGPU_PLACE && type != FPGA_PLACE);

    xmlFree(numStr);
    xmlFree(didStr);
    xmlFree(typeStr);
    xmlFree(sizeStr);
    xmlFree(unitSize);
    xmlFree(info);

    place_t *pl = (place_t *) malloc(sizeof(place_t));
    HASSERT(pl);
    memset(pl, 0x00, sizeof(place_t));
    pl->id = num;
    pl->did = did;
    pl->type = type;
    pl->psize = (sizeStr == NULL) ? 0 : atoi((char *)sizeStr);
    pl->unitSize = (unitSize == NULL) ? 0 : atoi((char *)unitSize);

    xmlNode *child = plNode->xmlChildrenNode;

    place_t *pllast = NULL;
    hclib_worker_state *wslast = NULL;

#ifdef HC_ASSERTION_CHECK
    unsigned nchildren = 0;
#endif
    while (child != NULL) {
        if (!xmlStrcmp(child->name, (const xmlChar *) "place")) {
            place_t *tmp = parse_place_element(child);
            tmp->parent = pl;
            if (pl->child == NULL) pl->child = tmp;
            else pllast->nnext = tmp;
            pllast = tmp;
#ifdef HC_ASSERTION_CHECK
            nchildren++;
#endif
        } else if (!xmlStrcmp(child->name, (const xmlChar *) "worker")) {
            hclib_worker_state *tmp = parse_worker_element(child);
            tmp->pl = pl;
            if (pl->workers == NULL) pl->workers = tmp;
            else wslast->next_worker = tmp;
            wslast = tmp;
#ifdef HC_ASSERTION_CHECK
            nchildren++;
#endif
        }
        child = child->next;
    }

    if (type == NVGPU_PLACE) HASSERT(nchildren == 0);

    return pl;
}

/*
 * Generates an in-memory representation of an HPT, and returns the first element in the top layer.
 *
 * In memory, an HPT is stored as something like:
 *
 *      + -> + -> +
 *      |         |
 *      |         |
 *      + -> +    + -> +
 *
 * with each layer consisting of a linked list of memory spaces/processing
 * units, and each of those nodes pointing down to the layer below it.
 */
place_t *parseHPTDoc(xmlNode *hptNode) {
    xmlChar *version = xmlGetProp(hptNode, xmlCharStrdup("version"));
    xmlChar *info = xmlGetProp(hptNode, xmlCharStrdup("info"));
    xmlFree(version);
    xmlFree(info);

    xmlNode *child = hptNode->xmlChildrenNode;

    place_t *hpt = NULL;
    place_t *pllast = NULL;

    // Iterative over top-level place nodes in the XML file
    while (child != NULL) {
        if (!xmlStrcmp(child->name, (const xmlChar *) "place")) {
            place_t *tmp = parse_place_element(child);
            tmp->parent = NULL;
            if (hpt == NULL) {
                hpt = tmp;
                pllast = tmp;
            } else {
                pllast->nnext = tmp;
                pllast = tmp;
            }
        }
        child = child->next;
    }
    return hpt;
}

typedef struct place_node {
    place_t *data;
    struct place_node *next;
} place_node_t;

typedef struct worker_node {
    hclib_worker_state *data;
    struct worker_node *next;
} worker_node_t;

/*
 * Generate two lists for all places and workers below the place pl in the HPT:
 * a list of places that are leaves in the tree (i.e. have no child places) and
 * a list of all workers in the HPT (which are implicitly leaves, by the
 * definition of a worker).
 *
 * This is done recursively to find all leaf places and workers in a global HPT.
 */
void find_leaf(place_t *pl, place_node_t **pl_list_cur,
               worker_node_t **wk_list_cur) {
    place_t *child = pl->child;
    if (child == NULL) {
        // Leaf, add it to the pl_list
        place_node_t *new_pl = (place_node_t *)malloc(sizeof(place_node_t));
        HASSERT(new_pl);
        memset(new_pl, 0x00, sizeof(place_node_t));
        new_pl->data = pl;
        (*pl_list_cur)->next = new_pl;
        *pl_list_cur = new_pl;
    } else {
        while (child != NULL) {
            find_leaf(child, pl_list_cur, wk_list_cur);
            child = child->nnext;
        }
    }

    hclib_worker_state *wk = pl->workers;
    while (wk != NULL) {
        // Add any workers to wk_list
        worker_node_t *new_ws = (worker_node_t *) malloc(sizeof(worker_node_t));
        HASSERT(new_ws);
        memset(new_ws, 0x00, sizeof(worker_node_t));
        new_ws->data = wk;
        (*wk_list_cur)->next = new_ws;
        *wk_list_cur = new_ws;
        wk = wk->next_worker;
    }
}

place_t *clonePlace(place_t *pl, int *num_pl, int *num_wk);
void setup_worker_hpt_path(hclib_worker_state *worker, place_t *pl);

/*
 * When we write HPT XML files manually, a count or num can be provided for each
 * place or worker node which reduces the need to write duplicate XML for
 * sibling XML nodes that are identical. For example, rather than having to
 * write:
 *
 *   <worker/>
 *   <worker/>
 *
 * to represent two cores sharing a place in the cache hierarchy, we can simply
 * write:
 *
 *   <worker num="2"/>
 *
 * The same goes for places. However, it is cleaner at runtime to store the HPT
 * in its fully expanded form, where every node has a num/count of 1. unrollHPT
 * handles unrolling or expanding worker and place nodes that have a num/count >
 * 1 by duplicating/cloning them and their subtree. At a high level, it starts
 * by expanding worker nodes, then leaf place nodes, then works recursively up
 * the HPT from the leaf place nodes and expands as it goes.
 *
 * At the end, it goes back over the HPT and re-assigns worker and place IDs to
 * be consistent.
 */
void unrollHPT(place_t *hpt, place_t *** all_places, int *num_pl, int *nproc,
               hclib_worker_state *** all_workers, int *num_wk) {
    int i;
    place_node_t leaf_places = {NULL, NULL};
    worker_node_t leaf_workers = {NULL, NULL};
    place_node_t *pl_list_cur = &leaf_places;
    worker_node_t *wk_list_cur = &leaf_workers;

    place_t *plp = hpt;
    /*
     * Iterate over all top-level places in the HPT, collecting all leaf places
     * and workers in this platform.
     */
    while (plp != NULL) {
        find_leaf (plp, &pl_list_cur, &wk_list_cur);
        plp = plp->nnext;
    }

    *num_wk = 0;
    *num_pl = 0;
    *nproc = 0;
    /*
     * Take any workers that were supposed with num > 1 in the HPT XML file and
     * "unroll" them, i.e. fully expand them into separate nodes in the HPT
     * tree so that all workers have num == 1. Do this for every single worker
     * in the HPT, discovered by find_leaf.
     */
    wk_list_cur = leaf_workers.next;
    while (wk_list_cur != NULL) {
        hclib_worker_state *ws = wk_list_cur->data;

        // num is read from the XML file and temporarily stored in the id field.
        int num = ws->id;
        for (i=0; i<num-1; i++) {
            hclib_worker_state *tmp = (hclib_worker_state *) malloc(
                                          sizeof(hclib_worker_state));
            HASSERT(tmp);
            memset(tmp, 0x00, sizeof(hclib_worker_state));
            tmp->pl = ws->pl;
            tmp->did = ws->did + num - i -
                       1; /* please note the way we add to the list, and the way we allocate did */
            tmp->next_worker = ws->next_worker;
            ws->next_worker = tmp;
        }
        (*nproc) += num;
        ws->pl->ndeques = ws->pl->ndeques + num;
        worker_node_t *tmpwknode = wk_list_cur;
        wk_list_cur = wk_list_cur->next;
        free(tmpwknode);
    }

    /* unroll places, we need to clone HPT subtree when we duplicate a place*/
    place_node_t *lastpl = pl_list_cur;
    pl_list_cur = leaf_places.next;
    while (pl_list_cur != NULL) {
        place_t *pl = pl_list_cur->data;

        /*
         * we first check whether this is the last child of its parent that
         * needs unrolling
         */
        place_t *parent = pl->parent;
        int unfinished = 0;
        if (parent != NULL) {
            place_t *pllast = parent->child;

            while (pllast != NULL) {
                /*
                 * id is cleared to -1 when we start processing a place node,
                 * check for places we haven't started processing.
                 */
                if (pllast->id != -1) unfinished ++;
                pllast = pllast->nnext;
            }
        }

        /* now do the clone and link them as sibling */
        (*num_pl) ++;

        // TODO This will add too many deques
        if (pl->parent != NULL) pl->parent->ndeques += pl->ndeques;
        int num = pl->id;
        pl->id = -1; /* clear this out, we only reset this after processing the whole HPT */

        for (i = 0; i < num - 1; i++) {
            place_t *clpl = clonePlace(pl, num_pl, nproc);
            // TODO This will add too many deques
            if (pl->parent != NULL) pl->parent->ndeques += pl->ndeques;

            /* now they are sibling */
            clpl->did = pl->did + num - i -
                        1; /* please note the way we add to the list, and the way we allocate did */
            clpl->nnext = pl->nnext;
            pl->nnext = clpl;
        }

        if (unfinished == 1) {/* the one we just unrolled should be this one */
            /*
             * If this is the last child place to be unrolled for a given
             * parent, add that parent to the work list being iterated over by
             * pl_list_cur so that we expand it at some point in the future.
             * This leads to a bottom-up traversal of the place tree, expanding
             * place nodes only after all of their children have already been
             * expanded.
             */
            place_node_t *new_pl = (place_node_t *)malloc(sizeof(place_node_t));
            HASSERT(new_pl);
            memset(new_pl, 0x00, sizeof(place_node_t));
            new_pl->data = parent;
            lastpl->next = new_pl;
            lastpl = new_pl;
        }

        // Move to next place in place list
        place_node_t *tmppl = pl_list_cur;
        pl_list_cur = tmppl->next;
        free(tmppl);
    }

    /* re-sequence the unrolled HPT with a breadth-first traversal */
    place_node_t *start = NULL;
    place_node_t *end = NULL;
    place_node_t *tmp;
    plp = hpt;

    *all_places = (place_t **)malloc(sizeof(place_t *) * (*num_pl));
    int num_dev_wk = 0;
    /* allocate worker objects for CPU workers */
    *all_workers = (hclib_worker_state **)malloc(sizeof(hclib_worker_state *) *
                   (*nproc));

    // Construct a new list of top-level places, starting with start.
    while (plp != NULL) {
        tmp= (place_node_t *) malloc(sizeof(place_node_t));
        tmp->data = plp;
        if (start == NULL) start = tmp;
        else end->next = tmp;
        end = tmp;
        plp->level = 1;
        plp = plp->nnext;
    }
    if (end != NULL) end->next = NULL;

    int wkid = 0;
    int plid = 0;
    while (start != NULL) {
        plp = start->data;
        if (plp->level == 0) {
            fprintf(stderr, "Place level is not set!\n");
            return;
        }
        (*all_places)[plid] = plp;
#ifdef TODO
        if (is_device_place(plp)) num_dev_wk++;
#endif
        plp->id = plid++;
        plp->nChildren = 0;
        place_t *child = plp->child;
        while (child != NULL) {
            tmp = (place_node_t *) malloc(sizeof(place_node_t));
            tmp->data = child;
            tmp->next = NULL;
            end->next = tmp;
            end = tmp;
            plp->nChildren++;
            child->level = plp->level + 1;
            child = child->nnext;
        }

        plp->children = (place_t **)malloc(sizeof(place_t *) * plp->nChildren);
        child = plp->child;
        for (i = 0; i < plp->nChildren; i++) {
            plp->children[i] = child;
            child = child->nnext;
        }

        hclib_worker_state *ws = plp->workers;
        while (ws != NULL) {
            (*all_workers)[wkid] = ws;
            ws->id = wkid ++;
            setup_worker_hpt_path(ws, plp);
            ws = ws->next_worker;
        }
        tmp = start->next;
        free(start);
        start = tmp;
    }
    if (num_dev_wk) {
        *num_wk = *nproc + num_dev_wk;
        hclib_worker_state **tmp = *all_workers;
        *all_workers = (hclib_worker_state **)malloc(sizeof(hclib_worker_state *) *
                       (*num_wk));
        memcpy(*all_workers, tmp, sizeof(hclib_worker_state *)*(*nproc));
        free(tmp);

        /* link the device workers with the place and the all_workers list */
        for (i = 0; i < num_dev_wk; i++) {
            hclib_worker_state *ws = (hclib_worker_state *) malloc(
                                         sizeof(hclib_worker_state));
            HASSERT(ws);
            memset(ws, 0x00, sizeof(hclib_worker_state));
            ws->id = *nproc + i;
            (*all_workers)[ws->id] = ws;
        }
#ifdef TODO
        if (num_dev_wk) {
            int index = 0;
            for (i = 0; i < *num_pl; i++) {
                place_t *dev_pl = ((*all_places)[i]);
                if (is_device_place(dev_pl)) {
                    dev_pl->workers = (*all_workers)[*nproc + index];
                    (*all_workers)[*nproc + index]->pl = dev_pl;
                    (*all_workers)[*nproc + index]->did = dev_pl->did;
                    setup_worker_hpt_path((*all_workers)[*nproc + index], dev_pl);
                    index++;
                }
            }
        }
#endif
    } else {
        *num_wk = *nproc;
    }
}

/*
 * Recursively copy a place and its entire subtree, incrementing num_pl every
 * time a new place is created and incrementing nproc every time a new worker is
 * created.
 */
place_t *clonePlace(place_t *pl, int *num_pl, int *nproc) {
    place_t *clone = (place_t *) malloc(sizeof(place_t));
    HASSERT(clone);
    memset(clone, 0x00, sizeof(place_t));
    clone->type = pl->type;
    clone->psize = pl->psize;
    clone->unitSize = pl->unitSize;
    clone->id = pl->id;
    clone->did = pl->did;
    clone->ndeques = pl->ndeques;
    clone->parent = pl->parent;
    place_t *child = pl->child;

    place_t *pllast = NULL;
    while (child != NULL) {
        place_t *tmp = clonePlace(child, num_pl, nproc);
        tmp->parent = clone;
        if (clone->child == NULL) clone->child = tmp;
        else pllast->nnext = tmp;
        pllast = tmp;

        child = child->nnext;
    }
    if (pllast != NULL) pllast->nnext = NULL;

    hclib_worker_state *ws = pl->workers;
    hclib_worker_state *wslast = NULL;

    while (ws != NULL) {
        hclib_worker_state *tmp = (hclib_worker_state *) malloc(sizeof(
                                      hclib_worker_state));
        HASSERT(tmp);
        memset(tmp, 0x00, sizeof(hclib_worker_state));
        tmp->pl = clone;
        tmp->did = ws->did;
        if (clone->workers == NULL) clone->workers = tmp;
        else wslast->next_worker = tmp;
        wslast = tmp;
        (*nproc) ++;
        ws = ws->next_worker;
    }
    if (wslast != NULL) wslast->next_worker = NULL;
    (*num_pl) ++;

    return clone;
}

void setup_worker_hpt_path(hclib_worker_state *worker, place_t *pl) {
    int i;
    if (pl == NULL) return;
    int level = pl->level;
    worker->hpt_path = (place_t **)malloc(sizeof(place_t *) * (level+2));
    worker->hpt_path[level+1] = pl;
    for (i = level; i > 0; i--) {
        if (pl == NULL) {
            fprintf(stderr, "Place level does not match with tree depth!\n");
            return;
        }
        worker->hpt_path[i] = pl;
        pl = pl->parent;
    }
    worker->hpt_path[0] = worker->hpt_path[1];
}

place_t *generate_fake_hpt(uint32_t num_workers, place_t *** all_places,
                           int *num_pl, int *nproc, hclib_worker_state ***all_workers, int *num_wk) {
    uint32_t i;
    place_t *pl = (place_t *)malloc(sizeof(place_t));
    HASSERT(pl);
    memset(pl, 0x00, sizeof(place_t));

    pl->id = 1;
    pl->type = MEM_PLACE;
    pl->ndeques = num_workers;

    hclib_worker_state *last = NULL;
    for (i = 0; i < num_workers; i++) {
        hclib_worker_state *worker = (hclib_worker_state *)malloc(sizeof(
                                         hclib_worker_state));
        memset(worker, 0x00, sizeof(hclib_worker_state));
        worker->id = 1;
        worker->pl = pl;
        if (pl->workers == NULL) pl->workers = worker;
        else last->next_worker = worker;
        last = worker;
    }

    return pl;
}

/*
 * The hierarchical place tree for a shared memory system is specified as an XML
 * file containing the hierarchy of memory spaces it contains. In general, this
 * hierarchy is multi-layered and may include a separate layer for system
 * memory, L3 caches, L2 caches, L1 caches, etc.
 *
 * Each layer is described by two properties: a type and a count/num. For
 * example, the type may be "mem" whose contents are explicitly managed by the
 * programmer, or a "cache" whose contents are automatically managed. The count
 * refers to the fan-out at a given level. For example, a machine with main
 * memory feeding to two sockets each with their own L3 cache would have a count
 * of 2 for the L3 layer.
 *
 * An example HPT specification is below. This specification describes a machine
 * with a single system memory, two L3 caches, each of which fans out to 6 L1/L2
 * caches.
 *
 * <HPT version="0.1" info="2 hex core2 Intel Westmere processors">
 *   <place num="1" type="mem">
 *     <place num="2" type="cache"> <!-- 2 sockets with common L3 in each -->
 *       <place num="6" type="cache"> <!-- 6 L2/L1 cache per socket -->
 *         <worker num="1"/>
 *       </place>
 *     </place>
 *   </place>
 * </HPT>
 *
 * While homogeneous systems will usually have this singly-nested structure, a
 * machine with multiple types of memory sitting below system memory may have
 * multiple elements at the same nesting level, e.g. both L3 and GPU device
 * memory at the same level in the hierarchy.
 *
 * read_hpt parses one of these XML files and produces the equivalent place
 * hierarchy, returning the root of that hierarchy. The schema of the HPT XML
 * file is stored in $HCLIB_HOME/hpt/hpt.dtd.
 */
place_t *read_hpt(place_t *** all_places, int *num_pl, int *nproc,
                  hclib_worker_state *** all_workers, int *num_wk) {
    const char *filename = getenv("HCLIB_HPT_FILE");
    place_t *hpt;
    if (filename == NULL) {
        const char *workers_str = getenv("HCLIB_WORKERS");
        uint32_t num_workers;
        if (workers_str) {
            num_workers = atoi(workers_str);
        } else {
            num_workers = sysconf(_SC_NPROCESSORS_ONLN);
            fprintf(stderr, "WARNING: HCLIB_WORKERS not provided, running with "
                    "default of %u\n", num_workers);
        }

        hpt = generate_fake_hpt(num_workers, all_places, num_pl, nproc,
                                all_workers, num_wk);
    } else {
        /* create a parser context */
        xmlParserCtxt *ctxt = xmlNewParserCtxt();
        if (ctxt == NULL) {
            fprintf(stderr, "Failed to allocate parser context\n");
            return NULL;
        }
        /* parse the file, activating the DTD validation option */
        xmlDoc *doc = xmlCtxtReadFile(ctxt, filename, NULL, XML_PARSE_DTDVALID);
        /* check if parsing succeeded */
        if (doc == NULL) {
            fprintf(stderr, "Failed to parse %s\n", filename);
            return NULL;
        }

        /* check if validation suceeded */
        if (ctxt->valid == 0) {
            fprintf(stderr, "Failed to validate %s\n", filename);
            return NULL;
        }

        xmlNode *root_element = xmlDocGetRootElement(doc);

        hpt = parseHPTDoc(root_element);

        /*free the document */
        xmlFreeDoc(doc);

        /* free up the parser context */
        xmlFreeParserCtxt(ctxt);
    }

    /*
     * This takes places which have num > 1 and workers that have num > 1 and
     * fully expand them in the place tree so that every node in the tree
     * corresponds to a single place/worker.
     */
    unrollHPT(hpt, all_places, num_pl, nproc, all_workers, num_wk);

    return hpt;
}

void free_hpt(place_t *hpt) {
    place_node_t *start = NULL;
    place_node_t *end = NULL;
    place_node_t *tmp;
    place_t *plp = hpt;

    while (plp != NULL) {
        tmp = (place_node_t *)  malloc(sizeof(place_node_t));
        tmp->data = plp;
        if (start == NULL)
            start = tmp;
        else
            end->next = tmp;
        end = tmp;
        plp = plp->nnext;
    }
    if (end != NULL) end->next = NULL;

    while (start != NULL) {
        plp = start->data;
        place_t *child = plp->child;
        while (child != NULL) {
            tmp = (place_node_t *) malloc(sizeof(place_node_t));
            tmp->data = child;
            tmp->next = NULL;
            end->next = tmp;
            end = tmp;
            child = child->nnext;
        }

        hclib_worker_state *ws = plp->workers;
        while (ws != NULL) {
            hclib_worker_state *tmpwk = ws;
            ws = ws->next_worker;
            free(tmpwk);
        }
        tmp = start->next;
        free(start);
        free(plp);
        start = tmp;
    }
}


