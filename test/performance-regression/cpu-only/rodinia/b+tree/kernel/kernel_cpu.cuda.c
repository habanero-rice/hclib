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

#include <omp.h>									// (in directory known to compiler)			needed by openmp
#include <stdlib.h>									// (in directory known to compiler)			needed by malloc
#include <stdio.h>									// (in directory known to compiler)			needed by printf, stderr
#include <assert.h>

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

class pragma93_omp_parallel_hclib_async {
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
    int* volatile keys;
    int* volatile h_keys;
    long* volatile offset;
    long* volatile h_offset;
    volatile long knodes_elem;
    record* volatile ans;
    record* volatile h_ans;
    record* volatile records;
    record* volatile h_records;

    public:
        pragma93_omp_parallel_hclib_async(int set_i,
                long set_maxheight,
                int set_thid,
                int set_threadsPerBlock,
                knode* set_knodes,
                long* set_currKnode,
                int set_bid,
                int* set_keys,
                long* set_offset,
                long set_knodes_elem,
                record* set_ans,
                record* set_records) {
            i = set_i;
            maxheight = set_maxheight;
            thid = set_thid;
            threadsPerBlock = set_threadsPerBlock;
            h_knodes = set_knodes;
            h_currKnode = set_currKnode;
            bid = set_bid;
            h_keys = set_keys;
            h_offset = set_offset;
            knodes_elem = set_knodes_elem;
            h_ans = set_ans;
            h_records = set_records;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        knodes = NULL;
        currKnode = NULL;
        keys = NULL;
        offset = NULL;
        ans = NULL;
        records = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 6, h_knodes, h_currKnode, h_keys, h_offset, h_ans, h_records);
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
            if (keys == NULL && (char *)h_keys >= (char *)host_allocations[i] && ((char *)h_keys - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_keys - (char *)host_allocations[i]);
                memcpy((void *)(&keys), (void *)(&tmp), sizeof(void *));
            }
            if (offset == NULL && (char *)h_offset >= (char *)host_allocations[i] && ((char *)h_offset - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_offset - (char *)host_allocations[i]);
                memcpy((void *)(&offset), (void *)(&tmp), sizeof(void *));
            }
            if (ans == NULL && (char *)h_ans >= (char *)host_allocations[i] && ((char *)h_ans - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_ans - (char *)host_allocations[i]);
                memcpy((void *)(&ans), (void *)(&tmp), sizeof(void *));
            }
            if (records == NULL && (char *)h_records >= (char *)host_allocations[i] && ((char *)h_records - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_records - (char *)host_allocations[i]);
                memcpy((void *)(&records), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(knodes || h_knodes == NULL);
        assert(currKnode || h_currKnode == NULL);
        assert(keys || h_keys == NULL);
        assert(offset || h_offset == NULL);
        assert(ans || h_ans == NULL);
        assert(records || h_records == NULL);

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

	}
            }
        }
};

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
 { const int niters = (count) - (0);
kernel_launcher(niters, pragma93_omp_parallel_hclib_async(i, maxheight, thid, threadsPerBlock, knodes, currKnode, bid, keys, offset, knodes_elem, ans, records));
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

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif

