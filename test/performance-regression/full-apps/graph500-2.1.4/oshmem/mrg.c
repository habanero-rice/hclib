#include "mrg.h"
#include "mrg_skip_matrices.h"
#include "mod_arith_64bit.h"

void mrg_apply_transition(const mrg_transition_matrix* mat,
        const mrg_state* st, mrg_state* r) {
    uint_fast32_t o1 = mod_mac_y(mod_mul(mat->d, st->z1), mod_mac4(0, mat->s, st->z2, mat->a, st->z3, mat->b, st->z4, mat->c, st->z5));
    uint_fast32_t o2 = mod_mac_y(mod_mac2(0, mat->c, st->z1, mat->w, st->z2), mod_mac3(0, mat->s, st->z3, mat->a, st->z4, mat->b, st->z5));
    uint_fast32_t o3 = mod_mac_y(mod_mac3(0, mat->b, st->z1, mat->v, st->z2, mat->w, st->z3), mod_mac2(0, mat->s, st->z4, mat->a, st->z5));
    uint_fast32_t o4 = mod_mac_y(mod_mac4(0, mat->a, st->z1, mat->u, st->z2, mat->v, st->z3, mat->w, st->z4), mod_mul(mat->s, st->z5));
    uint_fast32_t o5 = mod_mac2(mod_mac3(0, mat->s, st->z1, mat->t, st->z2, mat->u, st->z3), mat->v, st->z4, mat->w, st->z5);
    r->z1 = o1;
    r->z2 = o2;
    r->z3 = o3;
    r->z4 = o4;
    r->z5 = o5;
}

void mrg_step(const mrg_transition_matrix* mat, mrg_state* state) {
    mrg_apply_transition(mat, state, state);
}

static void mrg_orig_step(mrg_state* state) { /* Use original A, not fully optimized yet */
  uint_fast32_t new_elt = mod_mac_y(mod_mul_x(state->z1), state->z5);
  state->z5 = state->z4;
  state->z4 = state->z3;
  state->z3 = state->z2;
  state->z2 = state->z1;
  state->z1 = new_elt;
}


uint_fast32_t mrg_get_uint_orig(mrg_state* state) {
  mrg_orig_step(state);
  return state->z1;
}

void mrg_skip(mrg_state* state, uint_least64_t exponent_high,
        uint_least64_t exponent_middle, uint_least64_t exponent_low) {
  int byte_index;
  for (byte_index = 0; exponent_low; ++byte_index, exponent_low >>= 8) {
    uint_least8_t val = (uint_least8_t)(exponent_low & 0xFF);
    if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
  }
  for (byte_index = 8; exponent_middle; ++byte_index, exponent_middle >>= 8) {
    uint_least8_t val = (uint_least8_t)(exponent_middle & 0xFF);
    if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
  }
  for (byte_index = 16; exponent_high; ++byte_index, exponent_high >>= 8) {
    uint_least8_t val = (uint_least8_t)(exponent_high & 0xFF);
    if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
  }
}

