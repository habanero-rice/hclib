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
 * hcpp-mm.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Ported from Habanero-C
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_MM_H_
#define HCPP_MM_H_

namespace hcpp {

#ifdef __USE_HC_MM__
/*
 * This file is part of the Habanero-C implementation and
 * distributed under the Modified BSD License.
 * See LICENSE for details.
 *
 * The Habanero Team (hc-l@rice.edu)
 * Author: Yonghong Yan
 *
 * hc memory management, (reference cilk internal-malloc.c)
 *
 * This header is included from hc.h and should not be used directly by application code
 */

#define SYS_BUCKET -1

//__x86_64 specific
#define HC_CACHE_LINE 64

#define HC_MM_CANONICALIZE_MACRO(msize, n) \
		if (msize <= n && n >= HC_CACHE_LINE) return n;

static __inline__ int hc_mm_canonicalize(size_t msize) {
	HC_MM_CANONICALIZE_MACRO(msize, 16);
	HC_MM_CANONICALIZE_MACRO(msize, 32);
	HC_MM_CANONICALIZE_MACRO(msize, 64);
	HC_MM_CANONICALIZE_MACRO(msize, 128);
	HC_MM_CANONICALIZE_MACRO(msize, 256);
	HC_MM_CANONICALIZE_MACRO(msize, 512);
	HC_MM_CANONICALIZE_MACRO(msize, 1024);
	HC_MM_CANONICALIZE_MACRO(msize, 2048);
	HC_MM_CANONICALIZE_MACRO(msize, 4096);
	return SYS_BUCKET; /* keep gcc happy */
}

#define HC_MM_SIZE_TO_BUCKET(msize, n, bucket) \
		if (msize <= n && n >= HC_CACHE_LINE) return bucket;

static __inline__ int hc_mm_size_to_bucket(size_t msize) {
	HC_MM_SIZE_TO_BUCKET(msize, 16, 0);
	HC_MM_SIZE_TO_BUCKET(msize, 32, 1);
	HC_MM_SIZE_TO_BUCKET(msize, 64, 2);
	HC_MM_SIZE_TO_BUCKET(msize, 128, 3);
	HC_MM_SIZE_TO_BUCKET(msize, 256, 4);
	HC_MM_SIZE_TO_BUCKET(msize, 512, 5);
	HC_MM_SIZE_TO_BUCKET(msize, 1024, 6);
	HC_MM_SIZE_TO_BUCKET(msize, 2048, 7);
	HC_MM_SIZE_TO_BUCKET(msize, 4096, 8);
	return SYS_BUCKET; /* keep gcc happy */
}

#define HC_MM_BUCKET_TO_SIZE(b, n, bucket) \
		if (bucket == b) return n;

static __inline__ int hc_mm_bucket_to_size(int b) {
	HC_MM_BUCKET_TO_SIZE(b, 16, 0);
	HC_MM_BUCKET_TO_SIZE(b, 32, 1);
	HC_MM_BUCKET_TO_SIZE(b, 64, 2);
	HC_MM_BUCKET_TO_SIZE(b, 128, 3);
	HC_MM_BUCKET_TO_SIZE(b, 256, 4);
	HC_MM_BUCKET_TO_SIZE(b, 512, 5);
	HC_MM_BUCKET_TO_SIZE(b, 1024, 6);
	HC_MM_BUCKET_TO_SIZE(b, 2048, 7);
	HC_MM_BUCKET_TO_SIZE(b, 4096, 8);
	return -1;
}

#define HC_MM_THRESHOLD 4096
#define HC_MM_BUCKETS 9
#define HC_CACHE_LINE_PAD  char __dummy[HC_CACHE_LINE]
#define HC_CACHE_LINE_ALIGNED(decl) \
		decl  __attribute__((aligned(HC_CACHE_LINE)))

/* descriptor for an hc mm block, we currently have it as 8 bytes*/
typedef struct hc_mm_block {
	unsigned int bucket; /* 4 bytes */
	unsigned int msize; /* 4 bytes, for a max of 4Gb */
} hc_mm_block;

/* this has to be the sizeof(struct hc_mm_block) */
#define BLOCK_HEADER_SIZE sizeof(struct hc_mm_block)
#define BUCKET(ptr) ((hc_mm_block*) (ptr)) [-1].bucket
#define HC_MM_SIZE(ptr) ((hc_mm_block*) (ptr)) [-1].msize
#define BLOCK_MEM(ptr) ((void *) &((hc_mm_block*) (ptr)) [-1])

/* descriptor for an hc mm bucket */
typedef struct hc_mm_bucket {
	void *free_list; /* pointer to free list */
	int length; /* length of the free list */
	/*
	 * number of elements that can be freed before a global
	 * batch free is necessary.
	 */
	int count;
} hc_mm_bucket;

/* statistics for hc_mm, used mainly for debugging */
struct hc_mm_stats {
	int used; /* bytes used; the sum must be 0 at the end */
	int nmalloc; /* # malloc - # free; sum must be 0 at the end */
	int in_free_lists; /* bytes in free lists at the end of the execution */

	/* number of elements in each bucket */
	int length[HC_MM_BUCKETS];
	HC_CACHE_LINE_PAD; /* pad to cache line to avoid false sharing */
};

/* experiment the internal memory management that is based on Cilk internal-malloc.c */
typedef struct hc_mm_context {
	hc_mm_bucket global_buckets [HC_MM_BUCKETS];
	struct hc_mm_stats global_mm_stats;
	struct hc_mm_stats *worker_mm_stats;
	int im_allocated;
	int im_wasted;
	pthread_mutex_t global_buckets_lock;
	char *global_pool_begin;
	char *global_pool_end;
	char **alloc_list;
	int alloc_list_index;
	int alloc_list_size;

	FILE * stats_file = stderr;
} hc_mm_context;

#endif
}


}


#endif /* HCPP_MM_H_ */
