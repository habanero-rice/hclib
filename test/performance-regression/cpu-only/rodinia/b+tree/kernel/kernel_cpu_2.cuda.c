#include <stdio.h>
__device__ inline int hclib_get_current_worker() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template<class functor_type>
__global__ void wrapper_kernel(unsigned niters, functor_type functor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(tid);
    }
}
template<class functor_type>
static void kernel_launcher(unsigned niters, functor_type functor) {
    const int threads_per_block = 256;
    const int nblocks = (niters + threads_per_block - 1) / threads_per_block;
    functor.transfer_to_device();
    const unsigned long long start = capp_current_time_ns();
    wrapper_kernel<<<nblocks, threads_per_block>>>(niters, functor);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error while synchronizing kernel - %s\n", cudaGetErrorString(err));
        exit(2);
    }
    const unsigned long long end = capp_current_time_ns();
    fprintf(stderr, "CAPP %llu ns\n", end - start);
    functor.transfer_from_device();
}
#ifdef __cplusplus
#ifdef __CUDACC__
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

class pragma103_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    int i;
    volatile long maxheight;
    int thid;
    volatile int threadsPerBlock;
    knode* volatile knodes;
    knode* volatile h_knodes;
    long* volatile currKnode;
    long* volatile h_currKnode;
    int bid;
    int* volatile start;
    int* volatile h_start;
    volatile long knodes_elem;
    long* volatile offset;
    long* volatile h_offset;
    long* volatile lastKnode;
    long* volatile h_lastKnode;
    int* volatile end;
    int* volatile h_end;
    long* volatile offset_2;
    long* volatile h_offset_2;
    int* volatile recstart;
    int* volatile h_recstart;
    int* volatile reclength;
    int* volatile h_reclength;

    public:
        pragma103_omp_parallel_hclib_async(int set_i,
                long set_maxheight,
                int set_thid,
                int set_threadsPerBlock,
                knode* set_knodes,
                long* set_currKnode,
                int set_bid,
                int* set_start,
                long set_knodes_elem,
                long* set_offset,
                long* set_lastKnode,
                int* set_end,
                long* set_offset_2,
                int* set_recstart,
                int* set_reclength) {
            i = set_i;
            maxheight = set_maxheight;
            thid = set_thid;
            threadsPerBlock = set_threadsPerBlock;
            h_knodes = set_knodes;
            h_currKnode = set_currKnode;
            bid = set_bid;
            h_start = set_start;
            knodes_elem = set_knodes_elem;
            h_offset = set_offset;
            h_lastKnode = set_lastKnode;
            h_end = set_end;
            h_offset_2 = set_offset_2;
            h_recstart = set_recstart;
            h_reclength = set_reclength;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        knodes = NULL;
        currKnode = NULL;
        start = NULL;
        offset = NULL;
        lastKnode = NULL;
        end = NULL;
        offset_2 = NULL;
        recstart = NULL;
        reclength = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 9, h_knodes, h_currKnode, h_start, h_offset, h_lastKnode, h_end, h_offset_2, h_recstart, h_reclength);
        device_allocations = (void **)malloc(nallocations * sizeof(void *));
        for (i = 0; i < nallocations; i++) {
            err = cudaMalloc((void **)&device_allocations[i], host_allocation_sizes[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            err = cudaMemcpy((void *)device_allocations[i], (void *)host_allocations[i], host_allocation_sizes[i], cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            if (knodes == NULL && (char *)h_knodes >= (char *)host_allocations[i] && ((char *)h_knodes - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_knodes - (char *)host_allocations[i]);
                memcpy((void *)(&knodes), (void *)(&tmp), sizeof(void *));
            }
            if (currKnode == NULL && (char *)h_currKnode >= (char *)host_allocations[i] && ((char *)h_currKnode - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_currKnode - (char *)host_allocations[i]);
                memcpy((void *)(&currKnode), (void *)(&tmp), sizeof(void *));
            }
            if (start == NULL && (char *)h_start >= (char *)host_allocations[i] && ((char *)h_start - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_start - (char *)host_allocations[i]);
                memcpy((void *)(&start), (void *)(&tmp), sizeof(void *));
            }
            if (offset == NULL && (char *)h_offset >= (char *)host_allocations[i] && ((char *)h_offset - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_offset - (char *)host_allocations[i]);
                memcpy((void *)(&offset), (void *)(&tmp), sizeof(void *));
            }
            if (lastKnode == NULL && (char *)h_lastKnode >= (char *)host_allocations[i] && ((char *)h_lastKnode - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_lastKnode - (char *)host_allocations[i]);
                memcpy((void *)(&lastKnode), (void *)(&tmp), sizeof(void *));
            }
            if (end == NULL && (char *)h_end >= (char *)host_allocations[i] && ((char *)h_end - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_end - (char *)host_allocations[i]);
                memcpy((void *)(&end), (void *)(&tmp), sizeof(void *));
            }
            if (offset_2 == NULL && (char *)h_offset_2 >= (char *)host_allocations[i] && ((char *)h_offset_2 - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_offset_2 - (char *)host_allocations[i]);
                memcpy((void *)(&offset_2), (void *)(&tmp), sizeof(void *));
            }
            if (recstart == NULL && (char *)h_recstart >= (char *)host_allocations[i] && ((char *)h_recstart - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_recstart - (char *)host_allocations[i]);
                memcpy((void *)(&recstart), (void *)(&tmp), sizeof(void *));
            }
            if (reclength == NULL && (char *)h_reclength >= (char *)host_allocations[i] && ((char *)h_reclength - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_reclength - (char *)host_allocations[i]);
                memcpy((void *)(&reclength), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(knodes);
        assert(currKnode);
        assert(start);
        assert(offset);
        assert(lastKnode);
        assert(end);
        assert(offset_2);
        assert(recstart);
        assert(reclength);

    }

    void transfer_from_device() {
        cudaError_t err;
        int i;
        for (i = 0; i < nallocations; i++) {
            err = cudaMemcpy((void *)host_allocations[i], (void *)device_allocations[i], host_allocation_sizes[i], cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            err = cudaFree(device_allocations[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        }
    }

        __device__ void operator()(int bid) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {

		// process levels of the tree
		for(i = 0; i < maxheight; i++){

			// process all leaves at each level
			for(thid = 0; thid < threadsPerBlock; thid++){

				if((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid+1] > start[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if(knodes[currKnode[bid]].indices[thid] < knodes_elem){
						offset[bid] = knodes[currKnode[bid]].indices[thid];
					}
				}
				if((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid+1] > end[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if(knodes[lastKnode[bid]].indices[thid] < knodes_elem){
						offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
					}
				}

			}

			// set for next tree level
			currKnode[bid] = offset[bid];
			lastKnode[bid] = offset_2[bid];

		}

		// process leaves
		for(thid = 0; thid < threadsPerBlock; thid++){

			// Find the index of the starting record
			if(knodes[currKnode[bid]].keys[thid] == start[bid]){
				recstart[bid] = knodes[currKnode[bid]].indices[thid];
			}

		}

		// process leaves
		for(thid = 0; thid < threadsPerBlock; thid++){

			// Find the index of the ending record
			if(knodes[lastKnode[bid]].keys[thid] == end[bid]){
				reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid]+1;
			}

		}

	}
            }
        }
};

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
 { const int niters = (count) - (0);
kernel_launcher(niters, pragma103_omp_parallel_hclib_async(i, maxheight, thid, threadsPerBlock, knodes, currKnode, bid, start, knodes_elem, offset, lastKnode, end, offset_2, recstart, reclength));
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

}  // main

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
