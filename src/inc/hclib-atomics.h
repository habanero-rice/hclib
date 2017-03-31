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
 * hclib-atomics.h
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
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
