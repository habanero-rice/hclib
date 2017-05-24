/**
 * fcontext
 * version 1.64.0-beta1
 *
 *                Copyright Nick Vrvilo 2017.
 *            https://github.com/DaoWen/fcontext/
 * Distributed under the Boost Software License, Version 1.0.
 *     (See accompanying file LICENSE_1_0.txt or copy at
 *           http://www.boost.org/LICENSE_1_0.txt)
 *
 * A simple, high-performance library providing cooperative user-level threads,
 * or fibers, with a basic C-language API, derived from the fiber-context
 * implementation in the Boost.Context project, version 1.64.0-beta1.
 * Note that the version number of this fcontext library simply tracks the
 * corresponding Boost.Context version number.
 *
 * To compile, link against the corresponding versions of jump, make and ontop
 * assembly files for your platform from the Boost.Context GitHub repository:
 * https://github.com/boostorg/context/tree/boost-1.64.0-beta1/src/asm
 *
 * The function prototypes are based on those in fcontext.hpp, available at
 * which includes the following obligatory copyright notice:
 *
 *              Copyright Oliver Kowalke 2009.
 * Distributed under the Boost Software License, Version 1.0.
 *     (See accompanying file LICENSE_1_0.txt or copy at
 *           http://www.boost.org/LICENSE_1_0.txt)
 *
 * The original fcontext.hpp source file is available on GitHub:
 * https://github.com/boostorg/context/blob/boost-1.64.0-beta1/include/boost/context/detail/fcontext.hpp
 *
 * All code in this file can be considered a derivative of the sources
 * listed above, and is distributed under the same license terms.
 */

#ifndef FCONTEXT_H_
#define FCONTEXT_H_

#include <inttypes.h>
#include <signal.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef FCONTEXT_SIZE
// Default context size (used for fiber stacks)
#define FCONTEXT_SIZE (1 << 18)  // 256KiB
#endif

// Compare our fcontext stack size to the recommended signal stack size
#if FCONTEXT_SIZE < MINSIGSTKSZ
#warning "fcontext size is smaller than recommended for your platform."
#endif

// fcontext_t is a pointer to the saved context for a fiber.
// Details of the context structure are platform-dependent,
// so we consider the fcontext_t's backing structure opaque.
typedef struct fcontext_opaque_t *fcontext_t;

// fcontext_transfer_t contains a pointer to the previous context,
// as well as a generic data argument passed to the new context.
// Simply called "transfer_t" in the original source code.
typedef struct {
    fcontext_t prev_context;
    void *data;
} fcontext_transfer_t;

// fcontext_fn_t is the type-signature for an entry-point function of a fiber.
typedef void (*fcontext_fn_t)(fcontext_transfer_t);

// fcontext_ontop_fn_t is the type-signature for an ontop_fcontext call
typedef fcontext_transfer_t (*fcontext_ontop_fn_t)(fcontext_transfer_t);

typedef struct {
    fcontext_t context;
    uint8_t stack[];
} fcontext_state_t;

// Abstracting the memory management functions used to allocate and free the
// fcontext stack and state. It's probably better to use LD_PRELOAD to change
// the default allocator application-wide than to change these definitions.
#define FCONTEXT_ALLOC(nbytes) malloc(nbytes)
#define FCONTEXT_FREE(ptr) free(ptr)

// Actual stack size (less the fcontext_t pointer's footprint)
#define FCONTEXT_STACK_SIZE ((FCONTEXT_SIZE) - sizeof(fcontext_state_t))

// Prototype for native jump_fcontext routine from Boost.Context
extern fcontext_transfer_t jump_fcontext(fcontext_t const to, void *vp);

// Prototype for native jump_fcontext routine from Boost.Context
extern fcontext_t make_fcontext(void *sp, size_t size, fcontext_fn_t fn);

// Prototype for native jump_fcontext routine from Boost.Context
extern fcontext_transfer_t ontop_fcontext(fcontext_t const to, void *vp,
                                          fcontext_ontop_fn_t fn);

/**
 * Create a new fiber context that will use entry_fn as its entry point.
 * Note that entry_fn should use the noreturn specifier
 * since it should not return.
 * If the entry_fn does return, then the application (not just the thread)
 * terminates via a call to exit(0).
 */
static inline fcontext_state_t *fcontext_create(fcontext_fn_t entry_fn) {
    fcontext_state_t *state = FCONTEXT_ALLOC(FCONTEXT_SIZE);
    uint8_t *stack_top = state->stack + FCONTEXT_STACK_SIZE;
    state->context = make_fcontext(stack_top, FCONTEXT_STACK_SIZE, entry_fn);
    return state;
}

/**
 * Free a dead fiber's state (including its execution stack).
 */
static inline void fcontext_destroy(fcontext_state_t *ctx) {
    FCONTEXT_FREE(ctx);
}

/**
 * Create a proxy context for a non-fcontext thread or fiber
 * (e.g., the original context of a pthread).
 */
static inline fcontext_state_t *fcontext_create_proxy(void) {
    fcontext_state_t *state = FCONTEXT_ALLOC(sizeof(*state));
    return state;
}

/**
 * Free a proxy context state
 * (does not destroy the original thread's execution stack).
 */
static inline void fcontext_destroy_proxy(fcontext_state_t *proxy_ctx) {
    // no difference in freeing proxy contexts
    fcontext_destroy(proxy_ctx);
}

/**
 * Swap from the current fiber context to another context, denoted by `next`.
 *
 * If `next` is a new context (initialized via fcontext_create),
 * then the context is entered via the entry_fn, and the current context
 * is passed to the new context via the entry_fn's fcontext_transfer_t argument.
 *
 * If `next` is a context that was saved via another call to fcontext_swap,
 * then that context is entered by returning from its fcontext_swap call,
 * and the current context is passed via the fcontext_transfer_t return value.
 *
 * A generic data argument (`arg`) can be passed to the `next` context,
 * and then accessed via the `data` field in the resulting fcontext_transfer_t.
 */
static inline fcontext_transfer_t fcontext_swap(fcontext_t next, void *arg) {
    return jump_fcontext(next, arg);
}

/**
 * Similar to fcontext_swap, but stacks an additional call ON TOP OF
 * the already-saved context in `base`, and switches to that context.
 * The return value from `fn` will be passed as the fcontext_transfer_t
 * to the original `base` context.
 */
static inline fcontext_transfer_t fcontext_run_on(fcontext_t base, void *arg,
                                                  fcontext_ontop_fn_t fn) {
    return ontop_fcontext(base, arg, fn);
}

#endif  // FCONTEXT_H_
