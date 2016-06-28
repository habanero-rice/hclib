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
 * hashmap_extension.h
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

/*
 * Avoiding deadlock when attempting isolation on multiple objects  
 * ---------------------------------------------------------------
 *
 * Prior art: "A data-centric approach to synchronization"
 *             http://dl.acm.org/citation.cfm?id=2160913
 *
 * As described in above paper, we assign unique index to each object
 * that has registered for enabling isolation. We maintain only one
 * hashmap of these objects (as keys) in hclib during the entire execution.
 * Whenever object(s) request enabling isolation, a lock is acquired on the hashmap
 * before attempting to create a bucket for this key-value pair in hashmap.
 * Every time an entry is pushed to hashmap, the entry is assigned the current
 * value of mutex_lock_index and then mutex_lock_index is incremented. Hashmap 
 * lock is release once the object(s) are added to the hashmap.
 *
 * User is allowed to pass multiple objects for isolation:
 *      hclib::isolation(obj1, obj2, obj3, ..., objN, atomic_operation_lambda);
 *
 * The above code can lead to deadlock: Suppose thread T1 requests isolation of obj1 and then obj2; 
 * and thread T2 requests isolation on obj2 and then obj1. T1 locks obj1 and T2 locks obj2. Now
 * T1 will try to lock obj2 and T2 will try to lock obj1, leading to a deadlock.
 *
 * To avoid situations like above, we assign (as in above paper) 
 * each object an unique index (mutex_lock_index).
 * Hence, in above situation, both T1 and T2 would first check the index of obj1 and obj2.
 * Assuming index_obj1 < index_obj2, both T1 and T2 would first attempt lock on obj1
 * followed by obj2. This will avoid the deadlock described above.
 *
 */

static uint64_t mutex_lock_index = 0;

inline void assign_index(Entry** e) {
  (*e)->index = mutex_lock_index++;
}

inline void decrement_index() {
  // This is not allowed
}

/* 
 * Extra functions added to hashmap.c to support indexing:
 * inline Entry* hashmapGetEntry(Hashmap* map, void* key);
 * void* hashmapGetIndexKey(Hashmap* map, void* key, uint64_t* index);
 */

