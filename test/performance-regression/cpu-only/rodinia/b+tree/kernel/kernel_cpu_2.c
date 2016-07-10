#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <omp.h>									// (in directory known to compiler)
#include <stdlib.h>									// (in directory known to compiler)
#include <assert.h>
#include <stdio.h>

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "timer.h"					// (in directory provided here)	needed by timer

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_cpu_2.h"							// (in directory provided here)

//========================================================================================================================================================================================================200
//	PLASMAKERNEL_GPU
//========================================================================================================================================================================================================200

typedef struct _pragma103_omp_parallel {
    int thid;
    int bid;
    long long (*time0_ptr);
    long long (*time1_ptr);
    long long (*time2_ptr);
    int i;
    int (*threadsPerBlock_ptr);
    int (*cores_arg_ptr);
    knode (*(*knodes_ptr));
    long (*knodes_elem_ptr);
    int (*order_ptr);
    long (*maxheight_ptr);
    int (*count_ptr);
    long (*(*currKnode_ptr));
    long (*(*offset_ptr));
    long (*(*lastKnode_ptr));
    long (*(*offset_2_ptr));
    int (*(*start_ptr));
    int (*(*end_ptr));
    int (*(*recstart_ptr));
    int (*(*reclength_ptr));
 } pragma103_omp_parallel;

static void pragma103_omp_parallel_hclib_async(void *____arg, const int ___iter0);
void 
kernel_cpu_2(	int cores_arg,

				knode *knodes,
				long knodes_elem,

				int order,
				long maxheight,
				int count,

				long *currKnode,
				long *offset,
				long *lastKnode,
				long *offset_2,
				int *start,
				int *end,
				int *recstart,
				int *reclength)
{

	//======================================================================================================================================================150
	//	Variables
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;

	// common variables
	int i;

	time0 = get_time();

	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

    {
	time1 = get_time();

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;

	// process number of querries
 { 
pragma103_omp_parallel *new_ctx = (pragma103_omp_parallel *)malloc(sizeof(pragma103_omp_parallel));
new_ctx->thid = thid;
new_ctx->bid = bid;
new_ctx->time0_ptr = &(time0);
new_ctx->time1_ptr = &(time1);
new_ctx->time2_ptr = &(time2);
new_ctx->i = i;
new_ctx->threadsPerBlock_ptr = &(threadsPerBlock);
new_ctx->cores_arg_ptr = &(cores_arg);
new_ctx->knodes_ptr = &(knodes);
new_ctx->knodes_elem_ptr = &(knodes_elem);
new_ctx->order_ptr = &(order);
new_ctx->maxheight_ptr = &(maxheight);
new_ctx->count_ptr = &(count);
new_ctx->currKnode_ptr = &(currKnode);
new_ctx->offset_ptr = &(offset);
new_ctx->lastKnode_ptr = &(lastKnode);
new_ctx->offset_2_ptr = &(offset_2);
new_ctx->start_ptr = &(start);
new_ctx->end_ptr = &(end);
new_ctx->recstart_ptr = &(recstart);
new_ctx->reclength_ptr = &(reclength);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = count;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma103_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
free(new_ctx);
 } 

	time2 = get_time();
    }

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

	printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time2-time0) / 1000000);

} 
static void pragma103_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma103_omp_parallel *ctx = (pragma103_omp_parallel *)____arg;
    int thid; thid = ctx->thid;
    int bid; bid = ctx->bid;
    int i; i = ctx->i;
    do {
    bid = ___iter0;
{

		// process levels of the tree
		for(i = 0; i < (*(ctx->maxheight_ptr)); i++){

			// process all leaves at each level
			for(thid = 0; thid < (*(ctx->threadsPerBlock_ptr)); thid++){

				if(((*(ctx->knodes_ptr))[(*(ctx->currKnode_ptr))[bid]].keys[thid] <= (*(ctx->start_ptr))[bid]) && ((*(ctx->knodes_ptr))[(*(ctx->currKnode_ptr))[bid]].keys[thid+1] > (*(ctx->start_ptr))[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if((*(ctx->knodes_ptr))[(*(ctx->currKnode_ptr))[bid]].indices[thid] < (*(ctx->knodes_elem_ptr))){
						(*(ctx->offset_ptr))[bid] = (*(ctx->knodes_ptr))[(*(ctx->currKnode_ptr))[bid]].indices[thid];
					}
				}
				if(((*(ctx->knodes_ptr))[(*(ctx->lastKnode_ptr))[bid]].keys[thid] <= (*(ctx->end_ptr))[bid]) && ((*(ctx->knodes_ptr))[(*(ctx->lastKnode_ptr))[bid]].keys[thid+1] > (*(ctx->end_ptr))[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if((*(ctx->knodes_ptr))[(*(ctx->lastKnode_ptr))[bid]].indices[thid] < (*(ctx->knodes_elem_ptr))){
						(*(ctx->offset_2_ptr))[bid] = (*(ctx->knodes_ptr))[(*(ctx->lastKnode_ptr))[bid]].indices[thid];
					}
				}

			}

			// set for next tree level
			(*(ctx->currKnode_ptr))[bid] = (*(ctx->offset_ptr))[bid];
			(*(ctx->lastKnode_ptr))[bid] = (*(ctx->offset_2_ptr))[bid];

		}

		// process leaves
		for(thid = 0; thid < (*(ctx->threadsPerBlock_ptr)); thid++){

			// Find the index of the starting record
			if((*(ctx->knodes_ptr))[(*(ctx->currKnode_ptr))[bid]].keys[thid] == (*(ctx->start_ptr))[bid]){
				(*(ctx->recstart_ptr))[bid] = (*(ctx->knodes_ptr))[(*(ctx->currKnode_ptr))[bid]].indices[thid];
			}

		}

		// process leaves
		for(thid = 0; thid < (*(ctx->threadsPerBlock_ptr)); thid++){

			// Find the index of the ending record
			if((*(ctx->knodes_ptr))[(*(ctx->lastKnode_ptr))[bid]].keys[thid] == (*(ctx->end_ptr))[bid]){
				(*(ctx->reclength_ptr))[bid] = (*(ctx->knodes_ptr))[(*(ctx->lastKnode_ptr))[bid]].indices[thid] - (*(ctx->recstart_ptr))[bid]+1;
			}

		}

	} ;     } while (0);
}

 // main

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
