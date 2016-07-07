#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct _pragma123_omp_parallel {
    long long (*time0_ptr);
    long long (*time1_ptr);
    long long (*time2_ptr);
    long long (*time3_ptr);
    long long (*time4_ptr);
    double (*alpha_ptr);
    double (*a2_ptr);
    int i;
    int j;
    int k;
    int l;
    long first_i;
    FOUR_VECTOR (*rA);
    FOUR_VECTOR (*fA);
    int pointer;
    long first_j;
    FOUR_VECTOR (*rB);
    double (*qB);
    double r2;
    double u2;
    double fs;
    double vij;
    double fxij;
    double fyij;
    double fzij;
    THREE_VECTOR d;
    par_str (*par_ptr);
    dim_str (*dim_ptr);
    box_str (*(*box_ptr));
    FOUR_VECTOR (*(*rv_ptr));
    double (*(*qv_ptr));
    FOUR_VECTOR (*(*fv_ptr));
 } pragma123_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma123_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int l) {
        }
};

#else
static void pragma123_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
typedef struct _main_entrypoint_ctx {
    long long time0;
    long long time1;
    long long time2;
    long long time3;
    long long time4;
    double alpha;
    double a2;
    int i;
    int j;
    int k;
    int l;
    long first_i;
    FOUR_VECTOR (*rA);
    FOUR_VECTOR (*fA);
    int pointer;
    long first_j;
    FOUR_VECTOR (*rB);
    double (*qB);
    double r2;
    double u2;
    double fs;
    double vij;
    double fxij;
    double fyij;
    double fzij;
    THREE_VECTOR d;
    par_str par;
    dim_str dim;
    box_str (*box);
    FOUR_VECTOR (*rv);
    double (*qv);
    FOUR_VECTOR (*fv);
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    long long time0; time0 = ctx->time0;
    long long time1; time1 = ctx->time1;
    long long time2; time2 = ctx->time2;
    long long time3; time3 = ctx->time3;
    long long time4; time4 = ctx->time4;
    double alpha; alpha = ctx->alpha;
    double a2; a2 = ctx->a2;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int k; k = ctx->k;
    int l; l = ctx->l;
    long first_i; first_i = ctx->first_i;
    FOUR_VECTOR (*rA); rA = ctx->rA;
    FOUR_VECTOR (*fA); fA = ctx->fA;
    int pointer; pointer = ctx->pointer;
    long first_j; first_j = ctx->first_j;
    FOUR_VECTOR (*rB); rB = ctx->rB;
    double (*qB); qB = ctx->qB;
    double r2; r2 = ctx->r2;
    double u2; u2 = ctx->u2;
    double fs; fs = ctx->fs;
    double vij; vij = ctx->vij;
    double fxij; fxij = ctx->fxij;
    double fyij; fyij = ctx->fyij;
    double fzij; fzij = ctx->fzij;
    THREE_VECTOR d; d = ctx->d;
    par_str par; par = ctx->par;
    dim_str dim; dim = ctx->dim;
    box_str (*box); box = ctx->box;
    FOUR_VECTOR (*rv); rv = ctx->rv;
    double (*qv); qv = ctx->qv;
    FOUR_VECTOR (*fv); fv = ctx->fv;
{

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

 { 
pragma123_omp_parallel *new_ctx = (pragma123_omp_parallel *)malloc(sizeof(pragma123_omp_parallel));
new_ctx->time0_ptr = &(time0);
new_ctx->time1_ptr = &(time1);
new_ctx->time2_ptr = &(time2);
new_ctx->time3_ptr = &(time3);
new_ctx->time4_ptr = &(time4);
new_ctx->alpha_ptr = &(alpha);
new_ctx->a2_ptr = &(a2);
new_ctx->i = i;
new_ctx->j = j;
new_ctx->k = k;
new_ctx->l = l;
new_ctx->first_i = first_i;
new_ctx->rA = rA;
new_ctx->fA = fA;
new_ctx->pointer = pointer;
new_ctx->first_j = first_j;
new_ctx->rB = rB;
new_ctx->qB = qB;
new_ctx->r2 = r2;
new_ctx->u2 = u2;
new_ctx->fs = fs;
new_ctx->vij = vij;
new_ctx->fxij = fxij;
new_ctx->fyij = fyij;
new_ctx->fzij = fzij;
new_ctx->d = d;
new_ctx->par_ptr = &(par);
new_ctx->dim_ptr = &(dim);
new_ctx->box_ptr = &(box);
new_ctx->rv_ptr = &(rv);
new_ctx->qv_ptr = &(qv);
new_ctx->fv_ptr = &(fv);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = dim.number_boxes;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((dim.number_boxes) - (0), pragma123_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma123_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 }  // for l
    } ;     free(____arg);
}

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

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->time0 = time0;
new_ctx->time1 = time1;
new_ctx->time2 = time2;
new_ctx->time3 = time3;
new_ctx->time4 = time4;
new_ctx->alpha = alpha;
new_ctx->a2 = a2;
new_ctx->i = i;
new_ctx->j = j;
new_ctx->k = k;
new_ctx->l = l;
new_ctx->first_i = first_i;
new_ctx->rA = rA;
new_ctx->fA = fA;
new_ctx->pointer = pointer;
new_ctx->first_j = first_j;
new_ctx->rB = rB;
new_ctx->qB = qB;
new_ctx->r2 = r2;
new_ctx->u2 = u2;
new_ctx->fs = fs;
new_ctx->vij = vij;
new_ctx->fxij = fxij;
new_ctx->fyij = fyij;
new_ctx->fzij = fzij;
new_ctx->d = d;
new_ctx->par = par;
new_ctx->dim = dim;
new_ctx->box = box;
new_ctx->rv = rv;
new_ctx->qv = qv;
new_ctx->fv = fv;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);


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

}  

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma123_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma123_omp_parallel *ctx = (pragma123_omp_parallel *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int k; k = ctx->k;
    int l; l = ctx->l;
    long first_i; first_i = ctx->first_i;
    FOUR_VECTOR (*rA); rA = ctx->rA;
    FOUR_VECTOR (*fA); fA = ctx->fA;
    int pointer; pointer = ctx->pointer;
    long first_j; first_j = ctx->first_j;
    FOUR_VECTOR (*rB); rB = ctx->rB;
    double (*qB); qB = ctx->qB;
    double r2; r2 = ctx->r2;
    double u2; u2 = ctx->u2;
    double fs; fs = ctx->fs;
    double vij; vij = ctx->vij;
    double fxij; fxij = ctx->fxij;
    double fyij; fyij = ctx->fyij;
    double fzij; fzij = ctx->fzij;
    THREE_VECTOR d; d = ctx->d;
    hclib_start_finish();
    do {
    l = ___iter0;
{

		//------------------------------------------------------------------------------------------100
		//	home box - box parameters
		//------------------------------------------------------------------------------------------100

		first_i = (*(ctx->box_ptr))[l].offset;												// offset to common arrays

		//------------------------------------------------------------------------------------------100
		//	home box - distance, force, charge and type parameters from common arrays
		//------------------------------------------------------------------------------------------100

		rA = &(*(ctx->rv_ptr))[first_i];
		fA = &(*(ctx->fv_ptr))[first_i];

		//------------------------------------------------------------------------------------------100
		//	Do for the # of (home+neighbor) boxes
		//------------------------------------------------------------------------------------------100

		for (k=0; k<(1+(*(ctx->box_ptr))[l].nn); k++) 
		{

			//----------------------------------------50
			//	neighbor box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = l;													// set first box to be processed to home box
			}
			else{
				pointer = (*(ctx->box_ptr))[l].nei[k-1].number;							// remaining boxes are neighbor boxes
			}

			//----------------------------------------50
			//	neighbor box - box parameters
			//----------------------------------------50

			first_j = (*(ctx->box_ptr))[pointer].offset; 

			//----------------------------------------50
			//	neighbor box - distance, force, charge and type parameters
			//----------------------------------------50

			rB = &(*(ctx->rv_ptr))[first_j];
			qB = &(*(ctx->qv_ptr))[first_j];

			//----------------------------------------50
			//	Do for the # of particles in home box
			//----------------------------------------50

			for (i=0; i<NUMBER_PAR_PER_BOX; i=i+1){

				// do for the # of particles in current (home or neighbor) box
				for (j=0; j<NUMBER_PAR_PER_BOX; j=j+1){

					// // coefficients
					r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]); 
					u2 = (*(ctx->a2_ptr))*r2;
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

	} ;     } while (0);
    ; hclib_end_finish_nonblocking();

}

#endif
 // main

#ifdef __cplusplus
}
#endif
