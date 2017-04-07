/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HCLIB_ATOMICS_H_
#define HCLIB_ATOMICS_H_

#include <stdbool.h>

#ifdef HAVE_C11_STDATOMIC

#include <stdatomic.h>

static inline int _hclib_atomic_load_relaxed(_Atomic int *target) {
    return atomic_load_explicit(target, memory_order_relaxed);
}

static inline int _hclib_atomic_load_acquire(_Atomic int *target) {
    return atomic_load_explicit(target, memory_order_acquire);
}

static inline void _hclib_atomic_store_relaxed(_Atomic int *target, int value) {
    atomic_store_explicit(target, value, memory_order_relaxed);
}

static inline void _hclib_atomic_store_release(_Atomic int *target, int value) {
    atomic_store_explicit(target, value, memory_order_release);
}

static inline int _hclib_atomic_inc_relaxed(_Atomic int *target) {
    return atomic_fetch_add_explicit(target, 1, memory_order_relaxed) + 1;
}

static inline int _hclib_atomic_inc_acquire(_Atomic int *target) {
    return atomic_fetch_add_explicit(target, 1, memory_order_acquire) + 1;
}

static inline int _hclib_atomic_inc_release(_Atomic int *target) {
    return atomic_fetch_add_explicit(target, 1, memory_order_release) + 1;
}

static inline int _hclib_atomic_inc_acq_rel(_Atomic int *target) {
    return atomic_fetch_add_explicit(target, 1, memory_order_acq_rel) + 1;
}

static inline int _hclib_atomic_dec_relaxed(_Atomic int *target) {
    return atomic_fetch_sub_explicit(target, 1, memory_order_relaxed) - 1;
}

static inline int _hclib_atomic_dec_release(_Atomic int *target) {
    return atomic_fetch_sub_explicit(target, 1, memory_order_release) - 1;
}

static inline int _hclib_atomic_dec_acq_rel(_Atomic int *target) {
    return atomic_fetch_sub_explicit(target, 1, memory_order_acq_rel) - 1;
}

static inline bool _hclib_atomic_cas_acq_rel(_Atomic int *target, int expected, int desired) {
    return atomic_compare_exchange_strong_explicit(target, &expected, desired,
            memory_order_acq_rel, memory_order_relaxed);
}

#else /* !HAVE_C11_STDATOMIC */

#warning "Missing C11 atomics support, falling back to gcc atomics."

#ifndef _Atomic
#define _Atomic volatile
#endif

static inline int _hclib_atomic_load_relaxed(_Atomic int *target) {
    return *target;
}

static inline int _hclib_atomic_load_acquire(_Atomic int *target) {
    int res = *target;
    __sync_synchronize(); // acquire after read
    return res;
}

static inline void _hclib_atomic_store_relaxed(_Atomic int *target, int value) {
    *target = value;
}

static inline void _hclib_atomic_store_release(_Atomic int *target, int value) {
    __sync_synchronize(); // release before write
    *target = value;
}

static inline int _hclib_atomic_inc_relaxed(_Atomic int *target) {
    return __sync_add_and_fetch(target, 1);
}

static inline int _hclib_atomic_inc_acquire(_Atomic int *target) {
    return __sync_add_and_fetch(target, 1);
}

static inline int _hclib_atomic_inc_release(_Atomic int *target) {
    return __sync_add_and_fetch(target, 1);
}

static inline int _hclib_atomic_inc_acq_rel(_Atomic int *target) {
    return __sync_add_and_fetch(target, 1);
}

static inline int _hclib_atomic_dec_relaxed(_Atomic int *target) {
    return __sync_sub_and_fetch(target, 1);
}

static inline int _hclib_atomic_dec_release(_Atomic int *target) {
    return __sync_sub_and_fetch(target, 1);
}

static inline int _hclib_atomic_dec_acq_rel(_Atomic int *target) {
    return __sync_sub_and_fetch(target, 1);
}

static inline bool _hclib_atomic_cas_acq_rel(_Atomic int *target, int expected, int desired) {
    // NOTE - Clang 3.5 has a bug with __sync_bool_compare_and_swap:
    // https://bugs.llvm.org//show_bug.cgi?format=multiple&id=21499
    return __sync_val_compare_and_swap(target, expected, desired) == expected;
}

#endif /* HAVE_C11_STDATOMIC */

#endif /* HCLIB_ATOMICS_H_ */
