#include <string.h>

#include "hclib.h"
#include "hclib-rt.h"
#include "hclib-task.h"
#include "hclib-asyncStruct.h"
#include "hclib-finish.h"

#ifdef __cplusplus
extern "C" {
#endif

/*** START ASYNC IMPLEMENTATION ***/

void hclib_async(generic_framePtr fp, void *arg, hclib_promise_t** promise_list,
        struct _phased_t * phased_clause, place_t *place, int property) {
    HASSERT(property == 0);
    HASSERT(phased_clause == NULL);

    if (promise_list) {
        hclib_dependent_task_t *task = malloc(sizeof(hclib_dependent_task_t));
        task->async_task._fp = fp;
        task->async_task.is_asyncAnyType = 0;
        task->async_task.promise_list = NULL;
        task->async_task.args = arg;
        task->async_task.place = NULL;

        if (place) {
            spawn_await_at((hclib_task_t *)task, promise_list, place);
        } else {
            spawn_await((hclib_task_t *)task, promise_list);
        }
    } else {
        hclib_task_t *task = malloc(sizeof(hclib_task_t));
        task->_fp = fp;
        task->is_asyncAnyType = 0;
        task->promise_list = NULL;
        task->args = arg;
        task->place = NULL;

        if (place) {
            spawn_at_hpt(place, task);
        } else {
            spawn(task);
        }
    }
}

typedef struct _future_args_wrapper {
    hclib_promise_t event;
    futureFct_t fp;
    void *actual_in;
} future_args_wrapper;

static void future_caller(void *in) {
    future_args_wrapper *args = in;
    void *user_result = (args->fp)(args->actual_in);
    hclib_promise_put(&args->event, user_result);
}

hclib_promise_t *hclib_async_future(futureFct_t fp, void *arg,
        hclib_promise_t** promise_list, struct _phased_t * phased_clause,
        place_t *place, int property) {
    future_args_wrapper *wrapper = malloc(sizeof(future_args_wrapper));
    hclib_promise_init(&wrapper->event);
    wrapper->fp = fp;
    wrapper->actual_in = arg;
    hclib_async(future_caller, wrapper, promise_list, phased_clause, place,
            property);

    return (hclib_promise_t *)wrapper;
}

/*** END ASYNC IMPLEMENTATION ***/

/*** START FORASYNC IMPLEMENTATION ***/

#define DEBUG_FORASYNC 0

forasync1D_task_t * allocate_forasync1D_task() {
    forasync1D_task_t *forasync_task = (forasync1D_task_t *) malloc(
            sizeof(forasync1D_task_t));
    HASSERT(forasync_task && "malloc failed");
    forasync_task->forasync_task.place = NULL;
    return forasync_task;
}

forasync2D_task_t * allocate_forasync2D_task() {
    forasync2D_task_t * forasync_task = (forasync2D_task_t *) malloc(
            sizeof(forasync2D_task_t));
    HASSERT(forasync_task && "malloc failed");
    forasync_task->forasync_task.place = NULL;
    return forasync_task;
}

forasync3D_task_t * allocate_forasync3D_task() {
    forasync3D_task_t * forasync_task = (forasync3D_task_t *) malloc(
            sizeof(forasync3D_task_t));
    HASSERT(forasync_task && "malloc failed");
    forasync_task->forasync_task.place = NULL;
    return forasync_task;
}

void forasync1D_runner(void * forasync_arg) {
    forasync1D_t * forasync = (forasync1D_t *) forasync_arg;
    hclib_task_t *user = forasync->base.user;
    forasync1D_Fct_t user_fct_ptr = (forasync1D_Fct_t) user->_fp;
    void * user_arg = (void *) user->args;
    loop_domain_t loop0 = forasync->loop0;
    int i=0;
    for(i=loop0.low; i<loop0.high; i+=loop0.stride) {
        (*user_fct_ptr)(user_arg, i);
    }
}

void forasync2D_runner(void * forasync_arg) {
    forasync2D_t * forasync = (forasync2D_t *) forasync_arg;
    hclib_task_t * user = *((hclib_task_t **) forasync_arg);
    forasync2D_Fct_t user_fct_ptr = (forasync2D_Fct_t) user->_fp;
    void * user_arg = (void *) user->args;
    loop_domain_t loop0 = forasync->loop0;
    loop_domain_t loop1 = forasync->loop1;
    int i=0,j=0;
    for(i=loop0.low; i<loop0.high; i+=loop0.stride) {
        for(j=loop1.low; j<loop1.high; j+=loop1.stride) {
            (*user_fct_ptr)(user_arg, i, j);
        }
    }
}

void forasync3D_runner(void * forasync_arg) {
    forasync3D_t * forasync = (forasync3D_t *) forasync_arg;
    hclib_task_t * user = *((hclib_task_t **) forasync_arg);
    forasync3D_Fct_t user_fct_ptr = (forasync3D_Fct_t) user->_fp;
    void * user_arg = (void *) user->args;
    loop_domain_t loop0 = forasync->loop0;
    loop_domain_t loop1 = forasync->loop1;
    loop_domain_t loop2 = forasync->loop2;
    int i=0,j=0,k=0;
    for(i=loop0.low; i<loop0.high; i+=loop0.stride) {
        for(j=loop1.low; j<loop1.high; j+=loop1.stride) {
            for(k=loop2.low; k<loop2.high; k+=loop2.stride) {
                (*user_fct_ptr)(user_arg, i, j, k);
            }
        }
    }
#if DEBUG_FORASYNC
    printf("forasync spawned %d\n", nb_spawn);
#endif
}

void forasync1D_recursive(void * forasync_arg) {
    forasync1D_t * forasync = (forasync1D_t *) forasync_arg;
    loop_domain_t loop0 = forasync->loop0;
    int high0 = loop0.high;
    int low0 = loop0.low;
    int stride0 = loop0.stride;
    int tile0 = loop0.tile;

    //split the range into two, spawn a new task for the first half and recurse on the rest  
    forasync1D_task_t * new_forasync_task = NULL;
    if((high0-low0) > tile0) {
        int mid = (high0+low0)/2;
        // upper-half
        new_forasync_task = allocate_forasync1D_task();
        new_forasync_task->forasync_task._fp = forasync1D_recursive;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        loop_domain_t new_loop0 = {mid, high0, stride0, tile0};
        new_forasync_task->def.loop0 = new_loop0;
        // update lower-half
        forasync->loop0.high = mid;
        // delegate scheduling to the underlying runtime

        spawn((hclib_task_t *)new_forasync_task);
        //continue to work on the half task 
        forasync1D_recursive(forasync_arg); 
    } else { 
        //compute the tile
        forasync1D_runner(forasync_arg);
    }
}

void forasync2D_recursive(void * forasync_arg) {
    forasync2D_t * forasync = (forasync2D_t *) forasync_arg;
    loop_domain_t loop0 = forasync->loop0;
    int high0 = loop0.high;
    int low0 = loop0.low;
    int stride0 = loop0.stride;
    int tile0 = loop0.tile;
    loop_domain_t loop1 = forasync->loop1;
    int high1 = loop1.high;
    int low1 = loop1.low;
    int stride1 = loop1.stride;
    int tile1 = loop1.tile;

    //split the range into two, spawn a new task for the first half and recurse on the rest  
    forasync2D_task_t * new_forasync_task = NULL;
    if((high0-low0) > tile0) {
        int mid = (high0+low0)/2;
        // upper-half
        new_forasync_task = allocate_forasync2D_task();
        new_forasync_task->forasync_task._fp = forasync2D_recursive;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        loop_domain_t new_loop0 = {mid, high0, stride0, tile0};;
        new_forasync_task->def.loop0 = new_loop0;
        new_forasync_task->def.loop1 = loop1;
        // update lower-half
        forasync->loop0.high = mid;
    } else if((high1-low1) > tile1) {
        int mid = (high1+low1)/2;
        // upper-half
        new_forasync_task = allocate_forasync2D_task();
        new_forasync_task->forasync_task._fp = forasync2D_recursive;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        new_forasync_task->def.loop0 = loop0;
        loop_domain_t new_loop1 = {mid, high1, stride1, tile1};
        new_forasync_task->def.loop1 = new_loop1;
        // update lower-half
        forasync->loop1.high = mid;
    }
    // recurse
    if(new_forasync_task != NULL) {
        // delegate scheduling to the underlying runtime
        //TODO can we make this a special async to avoid a get_current_async ?
        spawn((hclib_task_t*)new_forasync_task);
        //continue to work on the half task 
        forasync2D_recursive(forasync_arg); 
    } else { //compute the tile
        forasync2D_runner(forasync_arg);
    }
}

void forasync3D_recursive(void * forasync_arg) {
    forasync3D_t * forasync = (forasync3D_t *) forasync_arg;
    loop_domain_t loop0 = forasync->loop0;
    int high0 = loop0.high;
    int low0 = loop0.low;
    int stride0 = loop0.stride;
    int tile0 = loop0.tile;
    loop_domain_t loop1 = forasync->loop1;
    int high1 = loop1.high;
    int low1 = loop1.low;
    int stride1 = loop1.stride;
    int tile1 = loop1.tile;
    loop_domain_t loop2 = forasync->loop2;
    int high2 = loop2.high;
    int low2 = loop2.low;
    int stride2 = loop2.stride;
    int tile2 = loop2.tile;

    //split the range into two, spawn a new task for the first half and recurse on the rest  
    forasync3D_task_t * new_forasync_task = NULL;
    if((high0-low0) > tile0) {
        int mid = (high0+low0)/2;
        // upper-half
        new_forasync_task = allocate_forasync3D_task();
        new_forasync_task->forasync_task._fp = forasync3D_recursive;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        loop_domain_t new_loop0 = {mid, high0, stride0, tile0};
        new_forasync_task->def.loop0 = new_loop0;
        new_forasync_task->def.loop1 = loop1;
        new_forasync_task->def.loop2 = loop2;
        // update lower-half
        forasync->loop0.high = mid;
    } else if((high1-low1) > tile1) {
        int mid = (high1+low1)/2;
        // upper-half
        new_forasync_task = allocate_forasync3D_task();
        new_forasync_task->forasync_task._fp = forasync3D_recursive;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        new_forasync_task->def.loop0 = loop0;
        loop_domain_t new_loop1 = {mid, high1, stride1, tile1};
        new_forasync_task->def.loop1 = new_loop1;
        new_forasync_task->def.loop2 = loop2;
        // update lower-half
        forasync->loop1.high = mid;
    } else if((high2-low2) > tile2) {
        int mid = (high2+low2)/2;
        // upper-half
        new_forasync_task = allocate_forasync3D_task();
        new_forasync_task->forasync_task._fp = forasync3D_recursive;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        new_forasync_task->def.loop0 = loop0;
        new_forasync_task->def.loop1 = loop1;
        loop_domain_t new_loop2 = {mid, high2, stride2, tile2};
        new_forasync_task->def.loop2 = new_loop2;
        // update lower-half
        forasync->loop2.high = mid;
    }
    // recurse
    if(new_forasync_task != NULL) {
        // delegate scheduling to the underlying runtime
        //TODO can we make this a special async to avoid a get_current_async ?
        spawn((hclib_task_t*)new_forasync_task);
        //continue to work on the half task 
        forasync3D_recursive(forasync_arg); 
    } else { //compute the tile
        forasync3D_runner(forasync_arg);
    }
}

void forasync1D_flat(void * forasync_arg) {
    forasync1D_t * forasync = (forasync1D_t *) forasync_arg;
    loop_domain_t loop0 = forasync->loop0;
    int high0 = loop0.high;
    int stride0 = loop0.stride;
    int tile0 = loop0.tile;
    int nb_chunks = (int) (high0/tile0);
    int size = tile0*nb_chunks;
    int low0;
    for(low0 = loop0.low; low0<size; low0+=tile0) {
#if DEBUG_FORASYNC
        printf("Scheduling Task %d %d\n",low0,(low0+tile0));
#endif
        //TODO block allocation ?
        forasync1D_task_t * new_forasync_task = allocate_forasync1D_task();
        new_forasync_task->forasync_task._fp = forasync1D_runner;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        loop_domain_t new_loop0 = {low0, low0+tile0, stride0, tile0};
        new_forasync_task->def.loop0 = new_loop0;
        // printf("1promise_list %p\n",((async_task_t*)new_forasync_task));
        // printf("2promise_list %p\n",((async_task_t*)new_forasync_task)->def.promise_list);
        spawn((hclib_task_t*)new_forasync_task);
    }
    // handling leftover
    if (size < high0) {
#if DEBUG_FORASYNC
        printf("Scheduling Task %d %d\n",low0,high0);
#endif
        forasync1D_task_t * new_forasync_task = allocate_forasync1D_task();
        new_forasync_task->forasync_task._fp = forasync1D_runner;
        new_forasync_task->forasync_task.args = &(new_forasync_task->def);
        new_forasync_task->forasync_task.promise_list = NULL;
        new_forasync_task->def.base.user = forasync->base.user;
        loop_domain_t new_loop0 = {low0, high0, loop0.stride, loop0.tile};
        new_forasync_task->def.loop0 = new_loop0;
        spawn((hclib_task_t*)new_forasync_task);
    }
}

void forasync2D_flat(void * forasync_arg) {
    forasync2D_t * forasync = (forasync2D_t *) forasync_arg;
    loop_domain_t loop0 = forasync->loop0;
    loop_domain_t loop1 = forasync->loop1;
    int low0, low1;
    for(low0=loop0.low; low0<loop0.high; low0+=loop0.tile) {
        int high0 = (low0+loop0.tile)>loop0.high?loop0.high:(low0+loop0.tile);
#if DEBUG_FORASYNC
        printf("Scheduling Task Loop1 %d %d\n",low0,high0);
#endif
        for(low1=loop1.low; low1<loop1.high; low1+=loop1.tile) {
            int high1 = (low1+loop1.tile)>loop1.high?loop1.high:(low1+loop1.tile);
#if DEBUG_FORASYNC
            printf("Scheduling Task %d %d\n",low1,high1);
#endif
            forasync2D_task_t * new_forasync_task = allocate_forasync2D_task();
            new_forasync_task->forasync_task._fp = forasync2D_runner;
            new_forasync_task->forasync_task.args = &(new_forasync_task->def);
            new_forasync_task->forasync_task.promise_list = NULL;
            new_forasync_task->def.base.user = forasync->base.user;
            loop_domain_t new_loop0 = {low0, high0, loop0.stride, loop0.tile};
            new_forasync_task->def.loop0 = new_loop0;
            loop_domain_t new_loop1 = {low1, high1, loop1.stride, loop1.tile};
            new_forasync_task->def.loop1 = new_loop1;
            spawn((hclib_task_t*)new_forasync_task);
        }
    }
}

void forasync3D_flat(void * forasync_arg) {
    forasync3D_t * forasync = (forasync3D_t *) forasync_arg;
    loop_domain_t loop0 = forasync->loop0;
    loop_domain_t loop1 = forasync->loop1;
    loop_domain_t loop2 = forasync->loop2;
    int low0, low1, low2;
    for(low0=loop0.low; low0<loop0.high; low0+=loop0.tile) {
        int high0 = (low0+loop0.tile)>loop0.high?loop0.high:(low0+loop0.tile);
#if DEBUG_FORASYNC
        printf("Scheduling Task Loop1 %d %d\n",low0,high0);
#endif
        for(low1=loop1.low; low1<loop1.high; low1+=loop1.tile) {
            int high1 = (low1+loop1.tile)>loop1.high?loop1.high:(low1+loop1.tile);
#if DEBUG_FORASYNC
            printf("Scheduling Task Loop2 %d %d\n",low1,high1);
#endif
            for(low2=loop2.low; low2<loop2.high; low2+=loop2.tile) {
                int high2 = (low2+loop2.tile)>loop2.high?loop2.high:(low2+loop2.tile);
#if DEBUG_FORASYNC
                printf("Scheduling Task %d %d\n",low2,high2);
#endif
                forasync3D_task_t * new_forasync_task = allocate_forasync3D_task();
                new_forasync_task->forasync_task._fp = forasync3D_runner;
                new_forasync_task->forasync_task.args = &(new_forasync_task->def);
                new_forasync_task->forasync_task.promise_list = NULL;
                new_forasync_task->def.base.user = forasync->base.user;
                loop_domain_t new_loop0 = {low0, high0, loop0.stride, loop0.tile};
                new_forasync_task->def.loop0 = new_loop0;
                loop_domain_t new_loop1 = {low1, high1, loop1.stride, loop1.tile};
                new_forasync_task->def.loop1 = new_loop1;
                loop_domain_t new_loop2 = {low2, high2, loop2.stride, loop2.tile};
                new_forasync_task->def.loop2 = new_loop2;
                spawn((hclib_task_t*)new_forasync_task);
            }
        }
    }
}

static void forasync_internal(void* user_fct_ptr, void * user_arg,
        int dim, loop_domain_t * loop_domain, forasync_mode_t mode) {
    // All the sub-asyncs share async_def

    // The user loop code to execute
    hclib_task_t *user_def = (hclib_task_t *)malloc(sizeof(hclib_task_t));
    HASSERT(user_def);
    user_def->_fp = user_fct_ptr;
    user_def->args = user_arg;
    user_def->promise_list = NULL;
    user_def->place = NULL;

    HASSERT(dim>0 && dim<4);
    // TODO put those somewhere as static
    asyncFct_t fct_ptr_rec[3] = { forasync1D_recursive, forasync2D_recursive,
        forasync3D_recursive };
    asyncFct_t fct_ptr_flat[3] = { forasync1D_flat, forasync2D_flat,
        forasync3D_flat };
    asyncFct_t * fct_ptr = (mode == FORASYNC_MODE_RECURSIVE) ? fct_ptr_rec : fct_ptr_flat;
    if (dim == 1) {
        forasync1D_t forasync = {{user_def}, loop_domain[0]};
        (fct_ptr[dim-1])((void *) &forasync);
    } else if (dim == 2) {
        forasync2D_t forasync = {{user_def}, loop_domain[0], loop_domain[1]};
        (fct_ptr[dim-1])((void *) &forasync);
    } else if (dim == 3) {
        forasync3D_t forasync = {{user_def}, loop_domain[0], loop_domain[1], loop_domain[2]};
        (fct_ptr[dim-1])((void *) &forasync);
    }
}

void hclib_forasync(void* forasync_fct, void * argv, hclib_promise_t** promise_list,
        int dim, loop_domain_t *domain, forasync_mode_t mode) {
    HASSERT(promise_list == NULL && "Limitation: forasync does not support DDFs yet");

    forasync_internal(forasync_fct, argv, dim, domain, mode);
}

hclib_promise_t *hclib_forasync_future(void* forasync_fct, void * argv,
        hclib_promise_t **promise_list, int dim, loop_domain_t *domain,
        forasync_mode_t mode) {

    hclib_start_finish();
    hclib_forasync(forasync_fct, argv, promise_list, dim, domain, mode);
    return hclib_end_finish_nonblocking();
}


/*** END FORASYNC IMPLEMENTATION ***/

#ifdef __cplusplus
}
#endif
