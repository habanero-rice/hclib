#ifndef _MRG_H
#define _MRG_H

#include <stdint.h>

typedef struct mrg_state {
    uint_fast32_t z1, z2, z3, z4, z5;
} mrg_state;

typedef struct mrg_transition_matrix {
  uint_fast32_t s, t, u, v, w;
  /* Cache for other parts of matrix (see mrg_update_cache function)     */
  uint_fast32_t a, b, c, d;
} mrg_transition_matrix;

extern void mrg_step(const mrg_transition_matrix* mat, mrg_state* state);
extern uint_fast32_t mrg_get_uint_orig(mrg_state* state);
extern void mrg_skip(mrg_state* state, uint_least64_t exponent_high,
        uint_least64_t exponent_middle, uint_least64_t exponent_low);

#endif
