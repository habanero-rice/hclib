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
 * hcpp-atomics.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_ATOMICS_H_
#define HCPP_ATOMICS_H_

namespace hcpp {

#if defined __x86_64 || __i686__

#define HC_CACHE_LINE 64


static __inline__ int hc_atomic_inc(volatile int *ptr) {
    unsigned char c;
    __asm__ __volatile__(

            "lock       ;\n"
            "incl %0; sete %1"
            : "+m" (*(ptr)), "=qm" (c)
              : : "memory"
    );
    return c!= 0;
}

/* return 1 if the *ptr becomes 0 after decremented, otherwise return 0
 */
static __inline__ int hc_atomic_dec(volatile int *ptr) {
    unsigned char rt;
    __asm__ __volatile__(
            "lock;\n"
            "decl %0; sete %1"
            : "+m" (*(ptr)), "=qm" (rt)
              : : "memory"
    );
    return rt != 0;
}

static __inline__ void hc_mfence() {
        __asm__ __volatile__("mfence":: : "memory");
}

/*
 * if (*ptr == ag) { *ptr = x, return 1 }
 * else return 0;
 */
static __inline__ int hc_cas(volatile int *ptr, int ag, int x) {
        int tmp;
        __asm__ __volatile__("lock;\n"
                             "cmpxchgl %1,%3"
                             : "=a" (tmp) /* %0 EAX, return value */
                             : "r"(x), /* %1 reg, new value */
                               "0" (ag), /* %2 EAX, compare value */
                               "m" (*(ptr)) /* %3 mem, destination operand */
                             : "memory" /*, "cc" content changed, memory and cond register */
                );
        return tmp == ag;
}

#endif /* __x86_64 */

}

#endif /* HCPP_ATOMICS_H_ */
