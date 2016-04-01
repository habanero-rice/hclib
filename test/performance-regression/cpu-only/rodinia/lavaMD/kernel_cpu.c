#include "hclib.h"
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

typedef struct _kernel_cpu111 {
    par_str par;
    dim_str dim;
    box_str *box;
    FOUR_VECTOR *rv;
    double *qv;
    FOUR_VECTOR *fv;
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
    FOUR_VECTOR *rA;
    FOUR_VECTOR *fA;
    int pointer;
    long first_j;
    FOUR_VECTOR *rB;
    double *qB;
    double r2;
    double u2;
    double fs;
    double vij;
    double fxij;
    double fyij;
    double fzij;
    THREE_VECTOR d;
 } kernel_cpu111;

static void kernel_cpu111_hclib_async(void *____arg, const int ___iter);void  kernel_cpu(	par_str par, 
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

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	 { 
kernel_cpu111 *ctx = (kernel_cpu111 *)malloc(sizeof(kernel_cpu111));
ctx->par = par;
ctx->dim = dim;
ctx->box = box;
ctx->rv = rv;
ctx->qv = qv;
ctx->fv = fv;
ctx->time0 = time0;
ctx->time1 = time1;
ctx->time2 = time2;
ctx->time3 = time3;
ctx->time4 = time4;
ctx->alpha = alpha;
ctx->a2 = a2;
ctx->i = i;
ctx->j = j;
ctx->k = k;
ctx->l = l;
ctx->first_i = first_i;
ctx->rA = rA;
ctx->fA = fA;
ctx->pointer = pointer;
ctx->first_j = first_j;
ctx->rB = rB;
ctx->qB = qB;
ctx->r2 = r2;
ctx->u2 = u2;
ctx->fs = fs;
ctx->vij = vij;
ctx->fxij = fxij;
ctx->fyij = fyij;
ctx->fzij = fzij;
ctx->d = d;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = dim.number_boxes;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)kernel_cpu111_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 }  // for l

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

} static void kernel_cpu111_hclib_async(void *____arg, const int ___iter) {
    kernel_cpu111 *ctx = (kernel_cpu111 *)____arg;
    par_str par; par = ctx->par;
    dim_str dim; dim = ctx->dim;
    box_str *box; box = ctx->box;
    FOUR_VECTOR *rv; rv = ctx->rv;
    double *qv; qv = ctx->qv;
    FOUR_VECTOR *fv; fv = ctx->fv;
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
    FOUR_VECTOR *rA; rA = ctx->rA;
    FOUR_VECTOR *fA; fA = ctx->fA;
    int pointer; pointer = ctx->pointer;
    long first_j; first_j = ctx->first_j;
    FOUR_VECTOR *rB; rB = ctx->rB;
    double *qB; qB = ctx->qB;
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
    l = ___iter;
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

	}    } while (0);
    ; hclib_end_finish();
}

 // main

#ifdef __cplusplus
}
#endif
