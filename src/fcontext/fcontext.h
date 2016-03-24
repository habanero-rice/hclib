/**
 * Link code against the correct versions of the jump*.S and make*.S files
 * from https://github.com/boostorg/context/tree/develop/src/asm.
 *
 * These function prototypes are based on those in fcontext.hpp, at
 * https://github.com/boostorg/context/blob/develop/include/boost/context,
 * which includes the following copyright notice:
 *
 *          Copyright Oliver Kowalke 2009.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 */

#include <stdbool.h>

typedef struct { void *sp; } fcontext_t;

/**
 * sp - top of the new stack (i.e., HIGHEST address!)
 * size - size of the stack
 * fn - function to call when starting the new context
 */
extern fcontext_t make_fcontext(void *sp, size_t size, void (*fn)(void*));

/**
 * ofc - old (current) context
 * nfc - new context (should already be set up)
 * vp  - if nfc was created by make_fcontext, then this is the argument to fn
 *       otherwise, this is the return value of jump_fcontext in nfc
 * preserve_fpu - should we save the floating point unit state?
 */
extern void *jump_fcontext(fcontext_t *ofc,
        fcontext_t nfc, void *vp, bool preserve_fpu);


