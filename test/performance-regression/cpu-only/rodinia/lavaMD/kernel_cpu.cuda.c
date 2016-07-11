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

#include <omp.h>									// (in path known to compiler)			needed by openmp
#include <stdlib.h>									// (in path known to compiler)			needed by malloc
#include <stdio.h>									// (in path known to compiler)			needed by printf
#include <math.h>									// (in path known to compiler)			needed by exp

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "main.h"								// (in the main program folder)	needed to recognized input variables

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "timer.h"					// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL_CPU FUNCTION HEADER
//======================================================================================================================================================150

#include "kernel_cpu.h"								// (in the current directory)

//========================================================================================================================================================================================================200
//	PLASMAKERNEL_GPU
//========================================================================================================================================================================================================200

class pragma123_omp_parallel_hclib_async {
    private:
    long first_i;
    box_str* volatile box;
    box_str* volatile h_box;
    int l;
    FOUR_VECTOR* rA;
    FOUR_VECTOR* h_rA;
    FOUR_VECTOR* volatile rv;
    FOUR_VECTOR* volatile h_rv;
    FOUR_VECTOR* fA;
    FOUR_VECTOR* h_fA;
    FOUR_VECTOR* volatile fv;
    FOUR_VECTOR* volatile h_fv;
    int k;
    int pointer;
    long first_j;
    FOUR_VECTOR* rB;
    FOUR_VECTOR* h_rB;
    double* qB;
    double* h_qB;
    double* volatile qv;
    double* volatile h_qv;
    int i;
    int j;
    double r2;
    double u2;
    volatile double a2;
    double vij;
    double fs;
    THREE_VECTOR d;
    double fxij;
    double fyij;
    double fzij;

    public:
        pragma123_omp_parallel_hclib_async(long set_first_i,
                box_str* set_box,
                int set_l,
                FOUR_VECTOR* set_rA,
                FOUR_VECTOR* set_rv,
                FOUR_VECTOR* set_fA,
                FOUR_VECTOR* set_fv,
                int set_k,
                int set_pointer,
                long set_first_j,
                FOUR_VECTOR* set_rB,
                double* set_qB,
                double* set_qv,
                int set_i,
                int set_j,
                double set_r2,
                double set_u2,
                double set_a2,
                double set_vij,
                double set_fs,
                THREE_VECTOR set_d,
                double set_fxij,
                double set_fyij,
                double set_fzij) {
            first_i = set_first_i;
            h_box = set_box;
            l = set_l;
            h_rA = set_rA;
            h_rv = set_rv;
            h_fA = set_fA;
            h_fv = set_fv;
            k = set_k;
            pointer = set_pointer;
            first_j = set_first_j;
            h_rB = set_rB;
            h_qB = set_qB;
            h_qv = set_qv;
            i = set_i;
            j = set_j;
            r2 = set_r2;
            u2 = set_u2;
            a2 = set_a2;
            vij = set_vij;
            fs = set_fs;
            d = set_d;
            fxij = set_fxij;
            fyij = set_fyij;
            fzij = set_fzij;

        }

    void transfer_to_device() {
        cudaError_t err;
        err = cudaMalloc((void **)&box, get_size_from_allocation(h_box));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)box, (void *)h_box, get_size_from_allocation(h_box), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&rA, get_size_from_allocation(h_rA));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)rA, (void *)h_rA, get_size_from_allocation(h_rA), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&rv, get_size_from_allocation(h_rv));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)rv, (void *)h_rv, get_size_from_allocation(h_rv), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&fA, get_size_from_allocation(h_fA));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)fA, (void *)h_fA, get_size_from_allocation(h_fA), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&fv, get_size_from_allocation(h_fv));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)fv, (void *)h_fv, get_size_from_allocation(h_fv), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&rB, get_size_from_allocation(h_rB));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)rB, (void *)h_rB, get_size_from_allocation(h_rB), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&qB, get_size_from_allocation(h_qB));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)qB, (void *)h_qB, get_size_from_allocation(h_qB), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMalloc((void **)&qv, get_size_from_allocation(h_qv));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)qv, (void *)h_qv, get_size_from_allocation(h_qv), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
    }

    void transfer_from_device() {
        cudaError_t err;
        err = cudaMemcpy((void *)h_box, (void *)box, get_size_from_allocation(h_box), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(box);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_rA, (void *)rA, get_size_from_allocation(h_rA), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(rA);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_rv, (void *)rv, get_size_from_allocation(h_rv), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(rv);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_fA, (void *)fA, get_size_from_allocation(h_fA), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(fA);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_fv, (void *)fv, get_size_from_allocation(h_fv), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(fv);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_rB, (void *)rB, get_size_from_allocation(h_rB), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(rB);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_qB, (void *)qB, get_size_from_allocation(h_qB), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(qB);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaMemcpy((void *)h_qv, (void *)qv, get_size_from_allocation(h_qv), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        err = cudaFree(qv);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
    }

        __device__ void operator()(int l) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {

		//------------------------------------------------------------------------------------------100
		//	home box - box parameters
		//------------------------------------------------------------------------------------------100

		first_i = box[l].offset;												// offset to common arrays

		//------------------------------------------------------------------------------------------100
		//	home box - distance, force, charge and type parameters from common arrays
		//------------------------------------------------------------------------------------------100

		rA = &rv[first_i];
		fA = &fv[first_i];

		//------------------------------------------------------------------------------------------100
		//	Do for the # of (home+neighbor) boxes
		//------------------------------------------------------------------------------------------100

		for (k=0; k<(1+box[l].nn); k++) 
		{

			//----------------------------------------50
			//	neighbor box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = l;													// set first box to be processed to home box
			}
			else{
				pointer = box[l].nei[k-1].number;							// remaining boxes are neighbor boxes
			}

			//----------------------------------------50
			//	neighbor box - box parameters
			//----------------------------------------50

			first_j = box[pointer].offset; 

			//----------------------------------------50
			//	neighbor box - distance, force, charge and type parameters
			//----------------------------------------50

			rB = &rv[first_j];
			qB = &qv[first_j];

			//----------------------------------------50
			//	Do for the # of particles in home box
			//----------------------------------------50

			for (i=0; i<NUMBER_PAR_PER_BOX; i=i+1){

				// do for the # of particles in current (home or neighbor) box
				for (j=0; j<NUMBER_PAR_PER_BOX; j=j+1){

					// // coefficients
					r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]); 
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2.*vij;
					d.x = rA[i].x  - rB[j].x; 
					d.y = rA[i].y  - rB[j].y; 
					d.z = rA[i].z  - rB[j].z; 
					fxij=fs*d.x;
					fyij=fs*d.y;
					fzij=fs*d.z;

					// forces
					fA[i].v +=  qB[j]*vij;
					fA[i].x +=  qB[j]*fxij;
					fA[i].y +=  qB[j]*fyij;
					fA[i].z +=  qB[j]*fzij;

				} // for j

			} // for i

		} // for k

	}
            }
        }
};

void  kernel_cpu(	par_str par, 
					dim_str dim,
					box_str* box,
					FOUR_VECTOR* rv,
					fp* qv,
					FOUR_VECTOR* fv)
{

	//======================================================================================================================================================150
	//	Variables
	//======================================================================================================================================================150

	// timer
	long long time0;

	time0 = get_time();

	// timer
	long long time1;
	long long time2;
	long long time3;
	long long time4;

	// parameters
	fp alpha;
	fp a2;

	// counters
	int i, j, k, l;

	// home box
	long first_i;
	FOUR_VECTOR* rA;
	FOUR_VECTOR* fA;

	// neighbor box
	int pointer;
	long first_j; 
	FOUR_VECTOR* rB;
	fp* qB;

	// common
	fp r2; 
	fp u2;
	fp fs;
	fp vij;
	fp fxij,fyij,fzij;
	THREE_VECTOR d;

	time1 = get_time();

	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	time2 = get_time();

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	alpha = par.alpha;
	a2 = 2.0*alpha*alpha;

	time3 = get_time();

{

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

 { const int niters = (dim.number_boxes) - (0);
kernel_launcher(niters, pragma123_omp_parallel_hclib_async(first_i, box, l, rA, rv, fA, fv, k, pointer, first_j, rB, qB, qv, i, j, r2, u2, a2, vij, fs, d, fxij, fyij, fzij));
 }  // for l
    }

	time4 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

	printf("%15.12f s, %15.12f % : CPU/MCPU: VARIABLES\n",				(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: INPUTS\n", 				(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time4-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time4-time0) / 1000000);

}  // main

// #ifdef __cplusplus
// }
// #endif
