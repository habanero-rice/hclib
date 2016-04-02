#include "hclib.h"
// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <omp.h>									// (in directory known to compiler)			needed by openmp
#include <stdlib.h>									// (in directory known to compiler)			needed by malloc
#include <stdio.h>									// (in directory known to compiler)			needed by printf, stderr

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "timer.h"					// (in directory provided here)

//========================================================================================================================================================================================================200
//	KERNEL_CPU FUNCTION
//========================================================================================================================================================================================================200

typedef struct _pragma84 {
    int cores_arg;
    record *records;
    knode *knodes;
    long knodes_elem;
    int order;
    long maxheight;
    int count;
    long *currKnode;
    long *offset;
    int *keys;
    record *ans;
    long long time0;
    long long time1;
    long long time2;
    int threadsPerBlock;
    int thid;
    int bid;
    int i;
 } pragma84;

static void pragma84_hclib_async(void *____arg, const int ___iter);
typedef struct _main_entrypoint_ctx {
    int cores_arg;
    record *records;
    knode *knodes;
    long knodes_elem;
    int order;
    long maxheight;
    int count;
    long *currKnode;
    long *offset;
    int *keys;
    record *ans;
    long long time0;
    long long time1;
    long long time2;
    int threadsPerBlock;
 } main_entrypoint_ctx;

static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int cores_arg; cores_arg = ctx->cores_arg;
    record *records; records = ctx->records;
    knode *knodes; knodes = ctx->knodes;
    long knodes_elem; knodes_elem = ctx->knodes_elem;
    int order; order = ctx->order;
    long maxheight; maxheight = ctx->maxheight;
    int count; count = ctx->count;
    long *currKnode; currKnode = ctx->currKnode;
    long *offset; offset = ctx->offset;
    int *keys; keys = ctx->keys;
    record *ans; ans = ctx->ans;
    long long time0; time0 = ctx->time0;
    long long time1; time1 = ctx->time1;
    long long time2; time2 = ctx->time2;
    int threadsPerBlock; threadsPerBlock = ctx->threadsPerBlock;
{

	time1 = get_time();

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;
	int i;

	// process number of querries
 { 
pragma84 *ctx = (pragma84 *)malloc(sizeof(pragma84));
ctx->cores_arg = cores_arg;
ctx->records = records;
ctx->knodes = knodes;
ctx->knodes_elem = knodes_elem;
ctx->order = order;
ctx->maxheight = maxheight;
ctx->count = count;
ctx->currKnode = currKnode;
ctx->offset = offset;
ctx->keys = keys;
ctx->ans = ans;
ctx->time0 = time0;
ctx->time1 = time1;
ctx->time2 = time2;
ctx->threadsPerBlock = threadsPerBlock;
ctx->thid = thid;
ctx->bid = bid;
ctx->i = i;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = count;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma84_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 

	time2 = get_time();
    } ; }

void 
kernel_cpu(	int cores_arg,

			record *records,
			knode *knodes,
			long knodes_elem,

			int order,
			long maxheight,
			int count,

			long *currKnode,
			long *offset,
			int *keys,
			record *ans)
{

	//======================================================================================================================================================150
	//	Variables
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;

	time0 = get_time();

	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->cores_arg = cores_arg;
ctx->records = records;
ctx->knodes = knodes;
ctx->knodes_elem = knodes_elem;
ctx->order = order;
ctx->maxheight = maxheight;
ctx->count = count;
ctx->currKnode = currKnode;
ctx->offset = offset;
ctx->keys = keys;
ctx->ans = ans;
ctx->time0 = time0;
ctx->time1 = time1;
ctx->time2 = time2;
ctx->threadsPerBlock = threadsPerBlock;
hclib_launch(main_entrypoint, ctx);
free(ctx);


	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

	printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time2-time0) / 1000000);

}  static void pragma84_hclib_async(void *____arg, const int ___iter) {
    pragma84 *ctx = (pragma84 *)____arg;
    int cores_arg; cores_arg = ctx->cores_arg;
    record *records; records = ctx->records;
    knode *knodes; knodes = ctx->knodes;
    long knodes_elem; knodes_elem = ctx->knodes_elem;
    int order; order = ctx->order;
    long maxheight; maxheight = ctx->maxheight;
    int count; count = ctx->count;
    long *currKnode; currKnode = ctx->currKnode;
    long *offset; offset = ctx->offset;
    int *keys; keys = ctx->keys;
    record *ans; ans = ctx->ans;
    long long time0; time0 = ctx->time0;
    long long time1; time1 = ctx->time1;
    long long time2; time2 = ctx->time2;
    int threadsPerBlock; threadsPerBlock = ctx->threadsPerBlock;
    int thid; thid = ctx->thid;
    int bid; bid = ctx->bid;
    int i; i = ctx->i;
    hclib_start_finish();
    do {
    bid = ___iter;
{

		// process levels of the tree
		for(i = 0; i < maxheight; i++){

			// process all leaves at each level
			for(thid = 0; thid < threadsPerBlock; thid++){

				// if value is between the two keys
				if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if(knodes[offset[bid]].indices[thid] < knodes_elem){
						offset[bid] = knodes[offset[bid]].indices[thid];
					}
				}

			}

			// set for next tree level
			currKnode[bid] = offset[bid];

		}

		//At this point, we have a candidate leaf node which may contain
		//the target record.  Check each key to hopefully find the record
		// process all leaves at each level
		for(thid = 0; thid < threadsPerBlock; thid++){

			if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
				ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
			}

		}

	} ;     } while (0);
    ; hclib_end_finish();
}



//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif

