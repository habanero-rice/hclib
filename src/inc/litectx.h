#ifndef _LITECTX_H_
#define _LITECTX_H_

#include "hclib_common.h"
#include "fcontext.h"
#include <string.h>
#include <sys/mman.h>

// #define OVERFLOW_PROTECT

#define LITECTX_FREE(ptr) free(ptr)
#define LITECTX_SIZE 0x40000 /* 256KB */
// #define LITECTX_SIZE 0x10000 /* 64KB */

#ifdef OVERFLOW_PROTECT
// This must be a multiple of the page size on this machine?
#define OVERFLOW_PADDING_SIZE 4096LU
#endif

typedef struct LiteCtxStruct {
    /*
     * If we're running in a mode where we want to make a best effort to detect
     * stack overflows by threads running on lightweight contexts, we create a
     * small buffer in front of the light context data structures and mprotect
     * it to PROT_NONE. This adds some overhead to context creation, but can be
     * useful for debugging odd behavior in HClib programs.
     */
#ifdef OVERFLOW_PROTECT
    char overflow_buffer[OVERFLOW_PADDING_SIZE];
#endif
    struct LiteCtxStruct *volatile prev;
    void *volatile arg1;
    void *volatile arg2;
    fcontext_t _fctx;
    char _stack[];
} LiteCtx;

#define LITECTX_STACK_SIZE ((LITECTX_SIZE) - sizeof(LiteCtx))

#ifdef OVERFLOW_PROTECT

extern int posix_memalign(void **memptr, size_t alignment, size_t size);

static inline void *LITECTX_ALLOC(size_t nbytes) {
    assert(nbytes == sizeof(LiteCtx) || nbytes == LITECTX_SIZE);

    void *ptr;
    const int err = posix_memalign(&ptr, (size_t)4096, nbytes);
    if (err != 0) {
        fprintf(stderr, "Error in posix_memalign\n");
        exit(1);
    }

    const int protect_err = mprotect(ptr, OVERFLOW_PADDING_SIZE, PROT_NONE);
    if (protect_err != 0) {
        perror("mprotect");
        exit(1);
    }

    printf("WARNING: Running in OVERFLOW_PROTECT mode to check for stack "
            "overflows, this will negatively impact performance.\n");
    printf("WARNING: Setting up PROT_NONE region at %p, length = %lu\n", ptr,
            OVERFLOW_PADDING_SIZE);

    return ptr;
}
#else
#define LITECTX_ALLOC(nbytes) malloc(nbytes)
#endif

static __inline__ LiteCtx *LiteCtx_create(void (*fn)(LiteCtx*)) {
    LiteCtx *ctx = (LiteCtx *)LITECTX_ALLOC(LITECTX_SIZE);
    if (!ctx) {
        fprintf(stderr, "Failed allocating litectx\n");
        exit(1);
    }
    char *const stack_top = ctx->_stack + LITECTX_STACK_SIZE;
    ctx->prev = NULL;
    ctx->arg1 = NULL;
    ctx->arg2 = NULL;
    ctx->_fctx = make_fcontext(stack_top, LITECTX_STACK_SIZE,
            (void (*)(void *))fn);

#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_create: %p, ctx size = %lu, stack size = %lu, "
            "stack top = %p, stack bottom = %p\n", ctx, sizeof(LiteCtx),
            LITECTX_STACK_SIZE, stack_top, ctx->_stack);
#endif
    return ctx;
}

static __inline__ void LiteCtx_destroy(LiteCtx *ctx) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_destroy: ctx=%p\n", ctx);
#endif

#ifdef OVERFLOW_PROTECT
    const int merr = mprotect(ctx, OVERFLOW_PADDING_SIZE,
            PROT_READ | PROT_WRITE);
    if (merr != 0) {
        perror("mprotect");
        exit(1);
    }
#endif
    LITECTX_FREE(ctx);
}

/**
 * Proxy contexts represent contexts that have an externally-managed
 * stack (e.g., the original context of a pthread).
 */
static __inline__ LiteCtx *LiteCtx_proxy_create(const char *lbl __attribute__((unused))) {
    LiteCtx *ctx = (LiteCtx *)LITECTX_ALLOC(sizeof(*ctx));
#ifdef OVERFLOW_PROTECT
    memset(((unsigned char *)ctx) + OVERFLOW_PADDING_SIZE, 0, sizeof(*ctx) - OVERFLOW_PADDING_SIZE);
#else
    memset(ctx, 0, sizeof(*ctx));
#endif

#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_proxy_create[%s]: %p\n", lbl, ctx);
#endif
    return ctx;
}

static __inline__ void LiteCtx_proxy_destroy(LiteCtx *ctx) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_proxy_destroy: ctx=%p\n", ctx);
#endif

#ifdef OVERFLOW_PROTECT
    const int merr = mprotect(ctx, OVERFLOW_PADDING_SIZE,
            PROT_READ | PROT_WRITE);
    if (merr != 0) {
        perror("mprotect");
        exit(1);
    }
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
        const char *lbl __attribute__((unused))) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_swap[%s]: current=%p(%p) next=%p(%p) on pthread "
            "%p\n", lbl, current, current->_fctx.sp, next, next->_fctx.sp,
            (void *)pthread_self());
#endif
    next->prev = current;
    LiteCtx *new_current = (LiteCtx *)jump_fcontext(&current->_fctx,
            next->_fctx, next, true);
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_swap: swapped in %p(%p) on pthread %p\n",
            new_current, new_current->_fctx.sp, (void *)pthread_self());
#endif
    return new_current;
}

#endif /* _LITECTX_H_ */
