#ifndef _LITECTX_H_
#define _LITECTX_H_

#include "hclib_common.h"
#include <stdlib.h>
#include "fcontext.h"
#include <string.h>

#define LITECTX_ALLOC(bytes) malloc(bytes)
#define LITECTX_FREE(ptr) free(ptr)
// #define LITECTX_FREE(ptr)
#define LITECTX_SIZE 0x10000 /* 64KiB */

typedef struct LiteCtxStruct {
    struct LiteCtxStruct *volatile prev;
    void *volatile arg;
    fcontext_t _fctx;
    char _stack[];
} LiteCtx;

#define LITECTX_STACK_SIZE ((LITECTX_SIZE) - sizeof(LiteCtx))

static __inline__ LiteCtx *LiteCtx_create(void (*fn)(LiteCtx*)) {
    LiteCtx *ctx = (LiteCtx *)LITECTX_ALLOC(LITECTX_SIZE);
    char *const stack_top = ctx->_stack + LITECTX_STACK_SIZE;
    ctx->prev = NULL;
    ctx->arg = NULL;
    ctx->_fctx = make_fcontext(stack_top, LITECTX_STACK_SIZE,
            (void (*)(void *))fn);
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_create: %p\n", ctx);
#endif
    return ctx;
}

static __inline__ void LiteCtx_destroy(LiteCtx *ctx) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_destroy: ctx=%p\n", ctx);
#endif
    LITECTX_FREE(ctx);
}

/**
 * Proxy contexts represent contexts that have an externally-managed
 * stack (e.g., the original context of a pthread).
 */
static __inline__ LiteCtx *LiteCtx_proxy_create(const char *lbl) {
    LiteCtx *ctx = (LiteCtx *)LITECTX_ALLOC(sizeof(*ctx));
    memset(ctx, 0, sizeof(*ctx));
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_proxy_create[%s]: %p\n", lbl, ctx);
#endif
    return ctx;
}

static __inline__ void LiteCtx_proxy_destroy(LiteCtx *ctx) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_proxy_destroy: ctx=%p\n", ctx);
#endif
    LITECTX_FREE(ctx);
}

/**
 * current - current context pointer
 * next - target context pointer
 * return - pointer to the current context,
 *   with the prev field set to the source context's pointer
 *
 * LiteCtx_swap is used to either swap in a newly created context on the current
 * thread (with an entrypoint function specified) or to switch back to a
 * previously created context.
 *
 * Swapping to a new context occurs in the following scenarios:
 *   1. When creating an initial new lite context as part of hclib_finalize,
 *      under which we perform the global hclib_end_finish.
 *   2. At the entrypoint of each worker thread, to create a lite context for
 *      all worker thread async and finishes to be performed under.
 *   3. From help_finish (called by end_finish), which creates a new lite
 *      context to switch to so that the current stack can be set aside as a
 *      continuation.
 *
 * Swapping back to a previously created context occurs in the following
 * scenarios:
 *   1. At the end of _hclib_finalize_ctx, as cleanup of the temporary lite
 *      context created for hclib_finalize.
 *   2. At the end of crt_work_loop, we switch back to the lite context that
 *      created the current lite context.
 *   3. In the escaping async created that is dependent on each finish, its only
 *      task is to swap out the current stack to the context for the
 *      continuation.
 *
 * NOTE: It is important to know that the boost::context library is designed so
 * that a fiber exiting its main entrypoint function will immediately call
 * exit(0). Therefore, it is important to be careful at the end of a fiber
 * entrypoint to always know what context to switch back to. If you do not swap
 * in another context, the entrypoint will exit and then your application will
 * exit silently with an exit code of zero. This can be hugely painful to debug.
 * It is good practice to end any function that acts as the entrypoint for a
 * fiber with an 'assert(0)' to ensure that if you do hit this case you get a
 * more sensible error message.
 */
static __inline__ LiteCtx *LiteCtx_swap(LiteCtx *current, LiteCtx *next,
        const char *lbl) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_swap[%s]: wid=%d current=%p(%p) next=%p(%p)\n",
            lbl, get_current_worker(), current, current->_fctx.sp, next,
            next->_fctx.sp);
#endif
    next->prev = current;
    LiteCtx *new_current = (LiteCtx *)jump_fcontext(&current->_fctx,
            next->_fctx, next, false);
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_swap: swapped back in %p(%p)\n", new_current,
            new_current->_fctx.sp);
#endif
    return new_current;
}

#endif /* _LITECTX_H_ */
