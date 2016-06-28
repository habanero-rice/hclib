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
 * hclib-isolated.c
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include <stdio.h>
#include "hclib-internal.h"
#include <stdbool.h>
#include "hashmap.h"

#define INITIAL_HASHMAP_SIZE 1048576
#define KNUTH_CONSTANT 2654435761
#define CHECK_RC(ret) assert((rc) != -1 && "pthread API call failed")
static Hashmap* isolated_map = NULL;


// TODO: Find a better hash function
// When using knuth constant in multiplication
// there might be integer overflow, so not using
// that at moment. The hashmap implementation
// copied from Android project operates on 
// int hash only, hence currently typecasting
// to int. Although its not safe to typecast pointers to int
static int hash(void * ptr) {
#if 0
  // Knuth's multiplicative hash
  // This hash function also being used in qthreads:
  // https://github.com/Qthreads/qthreads/blob/qtCnC/cnc/cnc.h
  return (((int)ptr) * KNUTH_CONSTANT);
#else
  return (int)ptr;
#endif
}

static bool equals(void* a, void* b) {
  return (a == b);
}

/*
 * Initialize the datastructures required for isolation implementation
 */
void init_isolation_datastructures() {
  isolated_map = hashmapCreate(INITIAL_HASHMAP_SIZE, hash, equals);
  assert(isolated_map);
}


/****************************************
 **** USER INTERFACES *****
 ***************************************/

void enable_isolation(const void * ptr) {
  int rc=0;
  pthread_mutex_t* mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
  assert(mutex && "malloc failed");
  CHECK_RC(rc=pthread_mutex_init(mutex, NULL));
  hashmapPut(isolated_map, ptr, mutex);
}

void enable_isolation_1d(const void * ptr, const int size) {
  int i;
  for(i=0; i<size; i++) {
    register_isolation_object(&(ptr[i]));
  }
}

void enable_isolation_2d(const void ** ptr, const int rows, const int col) {
  int i, j;
  for(i=0; i<rows; i++) {
    for(j=0; j<col; j++) {
      register_isolation_object(&(ptr[i][j]));
    }
  }
}

void disable_isolation(const void * ptr) {
  int rc=0;
  pthread_mutex_t* mutex = (pthread_mutex_t*) hashmapRemove(isolated_map, ptr);
  assert(mutex && "Failed to retrive value from hashmap");
  CHECK_RC(rc=pthread_mutex_destroy(mutex));
  free(mutex);
}

void disable_isolation_1d(const void * ptr, const int size) {
  int i;
  for(i=0; i<size; i++) {
    deregister_isolation_object(ptr+i);
  }
}

void disable_isolation_2d(const void ** ptr, const int rows, const int col) {
  int i, j;
  for(i=0; i<rows; i++) {
    for(j=0; j<col; j++) {
      deregister_isolation_object(&(ptr[i][j]));
    }
  }
}

void apply_isolated(const void* ptr) {
  int rc=0;
  pthread_mutex_t* mutex = (pthread_mutex_t*) hashmapGet(isolated_map, ptr);
  assert(mutex && "Failed to retrive value from hashmap");
  CHECK_RC(rc=pthread_mutex_lock(mutex));
}

void release_isolated(const void* ptr) {
  int rc=0;
  pthread_mutex_t* mutex = (pthread_mutex_t*) hashmapGet(isolated_map, ptr);
  assert(mutex && "Failed to retrive value from hashmap");
  CHECK_RC(rc=pthread_mutex_unlock(mutex));
}







