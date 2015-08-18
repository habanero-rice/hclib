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
 * hcpp-mm.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifdef __USE_HC_MM__
#include "hcpp-internal.h"
#define DEFAULT_ALLOC_LIST_SIZE   10
#define CONTEXT_MM(id) context->mm_context.id

namespace hcpp {

/* compute the length of a free list */
static int free_list_length(void *p) {
	int l = 0;
	while (p) {
		++l;
		p = ((void **) p)[0];
	}
	return l;
}

/* Free the allocation list */
static void free_alloc_list(hc_context * const context) {
	int i;

	for (i = 0; i <= CONTEXT_MM(alloc_list_index); i++)
		free(CONTEXT_MM(alloc_list)[i]);

	CONTEXT_MM(alloc_list_index) = -1;
}

/*************************************************************
 * global allocator
 *************************************************************/
#define HC_MM_CHUNK (32 * 1024)

static void extend_global_pool(hc_context * const context) {
	CONTEXT_MM(global_pool_begin) = (char*) malloc(HC_MM_CHUNK);
	check_log_die(CONTEXT_MM(global_pool_begin) == NULL, "internal malloc() out of memory\n");
	CONTEXT_MM(global_pool_end) = CONTEXT_MM(global_pool_begin) + HC_MM_CHUNK;
	CONTEXT_MM(im_allocated) += HC_MM_CHUNK;

	/* Now add this chunk to alloc list */
	CONTEXT_MM(alloc_list_index)++;

	/* First, enlarge alloc_list if it's already full */
	if (CONTEXT_MM(alloc_list_index) >= CONTEXT_MM(alloc_list_size)) {

		CONTEXT_MM(alloc_list) = (char**) realloc(CONTEXT_MM(alloc_list),
				(CONTEXT_MM(alloc_list_size) + DEFAULT_ALLOC_LIST_SIZE)
				* sizeof(char*));

		check_log_die(CONTEXT_MM(alloc_list) == NULL, "internal malloc() out of memory\n");

		CONTEXT_MM(alloc_list_size) += DEFAULT_ALLOC_LIST_SIZE;
	}

	CONTEXT_MM(alloc_list)[CONTEXT_MM(alloc_list_index)]
	                       = CONTEXT_MM(global_pool_begin);
}

/* the size is already canonicalized at this point */
static void *hc_mm_global_alloc(hc_context * const context, size_t size, int bucket) {
	void *mem;
	struct hc_mm_bucket *d;

	check_log_die(size > HC_MM_THRESHOLD, "threshold error");
	check_log_die(bucket >= HC_MM_BUCKETS, "buckets error");

	/* for debug */
	CONTEXT_MM(global_mm_stats).used += size;
	CONTEXT_MM(global_mm_stats).nmalloc += 1;

	d = CONTEXT_MM(global_buckets) + bucket;

	/* look at the global free list for this bucket */
	if ((mem = d->free_list)) {
		/* if there is a block in the free list */
		check_log_die(mem == NULL, "error");
		d->free_list = ((void **) mem)[0];
		d->count++;
	} else {
		/* allocate from the global pool */
		if (CONTEXT_MM(global_pool_begin) + size + BLOCK_HEADER_SIZE > CONTEXT_MM(global_pool_end)) {
			/*
			 * For now, we waste the fragment of pool.
			 * We could put the fragment in the free list instead...
			 */
			CONTEXT_MM(im_wasted) += (CONTEXT_MM(global_pool_end)
					- CONTEXT_MM(global_pool_begin));
			extend_global_pool(context);
		}
		mem = CONTEXT_MM(global_pool_begin);
		CONTEXT_MM(global_pool_begin) += (size + BLOCK_HEADER_SIZE);
		((hc_mm_block *) mem) ->bucket = bucket;
		((hc_mm_block *) mem) ->msize = size;
		mem = (void *) &((hc_mm_block *) mem)[1];
	}

	return mem;
}

/*
 * global_free simply returns the block to the free list
 */
static void hc_mm_global_free(hc_context * const context, void *p, size_t size, int bucket) {
	struct hc_mm_bucket *d;

	check_log_die(size > HC_MM_THRESHOLD, "");
	check_log_die(bucket >= HC_MM_BUCKETS, "");

	/* for debug */
	CONTEXT_MM(global_mm_stats).used -= size;
	CONTEXT_MM(global_mm_stats).nmalloc -= 1;

	d = CONTEXT_MM(global_buckets) + bucket;
	((void **) p)[0] = d->free_list;
	d->free_list = p;
	d->count--;
}

static void hc_mm_allocate_batch(hc_workerState * const ws, size_t size, int bucket) {
	int i;
	void *p;
	struct hc_mm_bucket *d = ws->buckets + bucket;

	hc_context * context = ws->context;

	pthread_mutex_lock(&CONTEXT_MM(global_buckets_lock));
	for (i = 0; i < context->options->alloc_batch_size / 2; ++i) {
		/* get it from global bucket */
		p = hc_mm_global_alloc(ws->context, size, bucket);

		/* put it in the local bucket */
		/* debug */
		CONTEXT_MM(worker_mm_stats)[ws->id].used -= size;
		CONTEXT_MM(worker_mm_stats)[ws->id].nmalloc -= 1;
		/* end of debug */
		((void **) p)[0] = d->free_list;
		d->free_list = p;
		d->count--;
	}
	pthread_mutex_unlock(&CONTEXT_MM(global_buckets_lock));
}

static void hc_mm_free_batch(hc_workerState * const ws, size_t size, int bucket) {
	int i;
	void *p;
	struct hc_mm_bucket *d = ws->buckets + bucket;

	hc_context * context = ws->context;
	pthread_mutex_lock(&CONTEXT_MM(global_buckets_lock));
	for (i = 0; i < context->options->alloc_batch_size / 2; ++i) {
		p = d->free_list;
		check_log_die(p == NULL, "local worker has no block to free to global bucket for bucket :%d\n", bucket);
		/* debug */
		CONTEXT_MM(worker_mm_stats)[ws->id].used += size;
		CONTEXT_MM(worker_mm_stats)[ws->id].nmalloc += 1;
		/* end of debug */
		/* remove it from the local bucket */
		d->free_list = ((void **) p)[0];
		d->count++;

		/* put it back to the global bucket */
		hc_mm_global_free(ws->context, p, size, bucket);
	}
	pthread_mutex_unlock(&CONTEXT_MM(global_buckets_lock));
}

void hc_mm_global_init_2(hc_context * const context) {
	int i;
	struct hc_mm_bucket *d;

	CONTEXT_MM(global_mm_stats).used = 0;
	CONTEXT_MM(global_mm_stats).nmalloc = 0;

	for (i = 0; i < context->nworkers; ++i) {
		CONTEXT_MM(worker_mm_stats)[i].used = 0;
		CONTEXT_MM(worker_mm_stats)[i].nmalloc = 0;

		/*
		 * bogus value.
		 * Will be updated by hc_mm_per_worker_cleanup
		 */
		CONTEXT_MM(worker_mm_stats)[i].in_free_lists = -1;
	}

	for (i = 0; i < HC_MM_BUCKETS; ++i) {
		d = CONTEXT_MM(global_buckets) + i;
		d->free_list = (void *) 0;
		d->count = context->options->alloc_batch_size; /* TODO: */
	}

	CONTEXT_MM(alloc_list_index) = -1;
	CONTEXT_MM(global_pool_begin) = (char*) 0;
	CONTEXT_MM(global_pool_end) = (char*) 0;
	CONTEXT_MM(im_allocated) = 0;
	CONTEXT_MM(im_wasted) = 0;
}

void hc_mm_global_init(hc_context * const context) {
	CONTEXT_MM(worker_mm_stats) = (struct hc_mm_stats *) malloc(context->nworkers * sizeof(struct hc_mm_stats));
	check_log_die(CONTEXT_MM(worker_mm_stats) == NULL, "Can't allocate hc_mm_stats\n");

	pthread_mutex_init(&CONTEXT_MM(global_buckets_lock), NULL);

	CONTEXT_MM(alloc_list) = (char**) malloc(DEFAULT_ALLOC_LIST_SIZE * sizeof(char*));
	check_log_die(CONTEXT_MM(alloc_list) == NULL, "Can't allocate alloc_list\n");

	CONTEXT_MM(alloc_list_size) = DEFAULT_ALLOC_LIST_SIZE;
}

void hc_mm_global_terminate(hc_context * const context) {
	pthread_mutex_destroy(&CONTEXT_MM(global_buckets_lock));
	free(CONTEXT_MM(worker_mm_stats));
	free(CONTEXT_MM(alloc_list));
}

void hc_mm_global_terminate_2(hc_context * const context) {
	free_alloc_list(context);
}

/*************************************************************
 * worker allocator
 *************************************************************/
 void *hc_mm_malloc(hc_workerState * const ws, size_t size) {
	if (size == 0) return NULL;
	int bucket;
	void *mem;
	struct hc_mm_bucket *d;
	hc_context * context = ws->context;

	/* debug */
	CONTEXT_MM(worker_mm_stats)[ws->id].used += size;
	CONTEXT_MM(worker_mm_stats)[ws->id].nmalloc += 1;
	/* end of debug */

	if (size > HC_MM_THRESHOLD) {
		HASSERT(false);
		hc_mm_block *blk = (hc_mm_block *) malloc(size + BLOCK_HEADER_SIZE);
		blk->bucket = SYS_BUCKET;
		blk->msize = size;
		return (void *) &blk[1];
	}

	size = hc_mm_canonicalize(size);
	bucket = hc_mm_size_to_bucket(size);
	d = ws->buckets + bucket;
	while (!((mem = d->free_list))) {
		hc_mm_allocate_batch(ws, size, bucket);
	}
	/* if there is a block in the free list */
	check_log_die(mem == NULL, "");
	d->free_list = ((void **) mem)[0];
	d->count++;

	return mem;
}

/*
 * FREE simply returns to the free list
 */
 void hc_mm_free(hc_workerState * const ws, void *p) {
	if (p == NULL) return;
	int bucket = BUCKET(p);
	int size = HC_MM_SIZE(p);
	struct hc_mm_bucket *d;

	hc_context * context = ws->context;

	/* debug */
	CONTEXT_MM(worker_mm_stats)[ws->id].used -= size;
	CONTEXT_MM(worker_mm_stats)[ws->id].nmalloc -= 1;
	/* end of debug */

	if (bucket == SYS_BUCKET) {
		HASSERT(false);
		free(BLOCK_MEM(p));
		return;
	}

	d = ws->buckets + bucket;

	while (d->count <= 0)
		hc_mm_free_batch(ws, size, bucket);

	((void **) p)[0] = d->free_list;
	d->free_list = p;
	d->count--;
}

void hc_mm_per_worker_init(hc_workerState * const ws) {
	int i;
	struct hc_mm_bucket *d;

	for (i = 0; i < HC_MM_BUCKETS; ++i) {
		d = ws->buckets + i;
		d->free_list = (void *) 0;
		d->count = ws->context->options->alloc_batch_size;
	}
}

void hc_mm_per_worker_cleanup(hc_workerState * const ws) {
	int in_free_lists;
	struct hc_mm_bucket *d;
	int i, k;

	hc_context * context = ws->context;
	in_free_lists = 0;
	for (i = 0; i < HC_MM_BUCKETS; ++i) {
		d = ws->buckets + i;
		k = free_list_length(d->free_list);
		check_log_die(d->count + k != context->options->alloc_batch_size,
				"HC runtime system detected an inconsistency in\n"
				"the free lists used internally for memory management.\n"
				"Either this is a hc bug (unlikely), or your program\n"
				"corrupted some hc data structures.\n");
		d->length = k;
		in_free_lists += hc_mm_bucket_to_size(i) * k;
		CONTEXT_MM(worker_mm_stats)[ws->id].length[i] = k;
	}

	CONTEXT_MM(worker_mm_stats)[ws->id].in_free_lists = in_free_lists;
}

/*************************************************************
 * Debugging and statistics
 *************************************************************/

void hc_mm_global_cleanup(hc_context * const context) {
	int i, total, tm, k;
	int in_free_lists;
	struct hc_mm_bucket *d;

	total = CONTEXT_MM(global_mm_stats).used;
	tm = CONTEXT_MM(global_mm_stats).nmalloc;
	for (i = 0; i < context->nworkers; ++i) {
		total += CONTEXT_MM(worker_mm_stats)[i].used;
		tm += CONTEXT_MM(worker_mm_stats)[i].nmalloc;
	}

	check_log_die(!(total == 0 && tm == 0),
			"HC runtime system detected a memory leak.\n"
			"Either this is a hc bug (unlikely), or your program\n"
			"exited without giving hc a chance to free its own\n"
			"memory.  Make sure you use Cilk_exit() and not exit().\n");

	in_free_lists = 0;
	for (i = 0; i < HC_MM_BUCKETS; ++i) {
		d = CONTEXT_MM(global_buckets) + i;
		k = free_list_length(d->free_list);
		check_log_die(d->count + k != context->options->alloc_batch_size,
				"HC runtime system detected an inconsistency in\n"
				"the free lists used internally for memory management.\n"
				"Either this is a cilk bug (unlikely), or your program\n"
				"corrupted some Cilk data structures.\n");
		d->length = k;
		in_free_lists += hc_mm_bucket_to_size(i) * k;
		CONTEXT_MM(global_mm_stats).length[i] = k;
	}
	CONTEXT_MM(global_mm_stats).in_free_lists = in_free_lists;
}

#define FIELD_DESC "%6d"
#define PN_DESC "%7d"
#define PN_NAME_DESC "%7s"

static void print_detailed_im_stats(hc_context *const context)
{
	int i, j;

	fprintf(CONTEXT_MM(stats_file),
			"\n"
			"Detailed count of elements in allocator buckets:\n"
			"\n");

	fprintf(CONTEXT_MM(stats_file), PN_NAME_DESC, "PN\\size");
	for (i = 0; i < HC_MM_BUCKETS; ++i) {
		fprintf(CONTEXT_MM(stats_file), FIELD_DESC,
				hc_mm_bucket_to_size(i));
	}
	fprintf(CONTEXT_MM(stats_file), "\n");

	fprintf(CONTEXT_MM(stats_file), PN_NAME_DESC, "Global");
	for (i = 0; i < HC_MM_BUCKETS; ++i) {
		fprintf(CONTEXT_MM(stats_file), FIELD_DESC,
				CONTEXT_MM(global_mm_stats).length[i]);
	}
	fprintf(CONTEXT_MM(stats_file), "\n");

	for (j = 0; j < context->nworkers; ++j) {
		fprintf(CONTEXT_MM(stats_file), PN_DESC, j);
		for (i = 0; i < HC_MM_BUCKETS; ++i) {
			fprintf(CONTEXT_MM(stats_file), FIELD_DESC,
					CONTEXT_MM(worker_mm_stats)[j].length[i]);
		}
		fprintf(CONTEXT_MM(stats_file), "\n");
	}

	fprintf(CONTEXT_MM(stats_file), "\n");
}

#define FORMAT "%14d\n"
void hc_mm_print_statistics(hc_context *const context)
{
	int i;
	int in_free_lists = 0;

	fprintf(CONTEXT_MM(stats_file),
			"\nSCHEDULER MEMORY STATISTICS:\n"
			"\n"
			"Total memory allocated for the scheduler: %d bytes\n"
			"Unfragmented memory in global pool: %ld bytes\n"
			"Memory wasted because of fragmentation: %d bytes\n"
			"Memory in free lists at end of execution (in bytes):\n",
			CONTEXT_MM(im_allocated),
			(long) (CONTEXT_MM(global_pool_end) -
					CONTEXT_MM(global_pool_begin)),
					CONTEXT_MM(im_wasted));

	fprintf(CONTEXT_MM(stats_file),
			"      PN     free-list\n"
			"  --------------------\n");

	fprintf(CONTEXT_MM(stats_file), "  Global" FORMAT,
			CONTEXT_MM(global_mm_stats).in_free_lists);
	in_free_lists += CONTEXT_MM(global_mm_stats).in_free_lists;

	for (i = 0; i < context->nworkers; ++i) {
		fprintf(CONTEXT_MM(stats_file), "     %3d" FORMAT, i,
				CONTEXT_MM(worker_mm_stats)[i].in_free_lists);
		in_free_lists += CONTEXT_MM(worker_mm_stats)[i].in_free_lists;
	}

	fprintf(CONTEXT_MM(stats_file),
			"  --------------------\n");

	fprintf(CONTEXT_MM(stats_file), "        " FORMAT, in_free_lists);

	print_detailed_im_stats(context);
}

void hc_mm_init(hc_context * context) {
	int i;
	/* hc mm init */
	hc_mm_global_init(context);
	hc_mm_global_init_2(context);

	/* init worker mm */
	for (i=0; i<context->nworkers; i++) {
		hc_workerState * ws = context->workers[i];
		hc_mm_per_worker_init(ws);
	}
}

void hc_mm_cleanup(hc_context * context) {
	int i;
	/* init worker mm */
	for (i = 0; i < context->nworkers; i++) {
		hc_workerState * ws = context->workers[i];
		hc_mm_per_worker_cleanup(ws);
	}
#ifdef HC_COMM_WORKER_STATS
	hc_mm_print_statistics(context);
#endif
	hc_mm_global_terminate_2(context);
	hc_mm_global_terminate(context);

}

}

#endif
