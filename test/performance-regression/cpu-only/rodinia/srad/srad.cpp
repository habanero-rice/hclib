#include "hclib.h"
// srad.cpp : Defines the entry point for the console application.
//

//#define OUTPUT


#define OPEN
#define	ITERATION
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void random_matrix(float *I, int rows, int cols);

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <no. of threads><lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<no. of threads>  - no. of threads\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}

typedef struct _main125 {
    int argc;
    char **argv;
    int rows;
    int cols;
    int size_I;
    int size_R;
    int niter;
    int iter;
    int k;
    float *I;
    float *J;
    float q0sqr;
    float sum;
    float sum2;
    float tmp;
    float meanROI;
    float varROI;
    float Jc;
    float G2;
    float L;
    float num;
    float den;
    float qsqr;
    int *iN;
    int *iS;
    int *jE;
    int *jW;
    float *dN;
    float *dS;
    float *dW;
    float *dE;
    int r1;
    int r2;
    int c1;
    int c2;
    float cN;
    float cS;
    float cW;
    float cE;
    float *c;
    float D;
    float lambda;
    int i;
    int j;
    int nthreads;
 } main125;

typedef struct _main158 {
    int argc;
    char **argv;
    int rows;
    int cols;
    int size_I;
    int size_R;
    int niter;
    int iter;
    int k;
    float *I;
    float *J;
    float q0sqr;
    float sum;
    float sum2;
    float tmp;
    float meanROI;
    float varROI;
    float Jc;
    float G2;
    float L;
    float num;
    float den;
    float qsqr;
    int *iN;
    int *iS;
    int *jE;
    int *jW;
    float *dN;
    float *dS;
    float *dW;
    float *dE;
    int r1;
    int r2;
    int c1;
    int c2;
    float cN;
    float cS;
    float cW;
    float cE;
    float *c;
    float D;
    float lambda;
    int i;
    int j;
    int nthreads;
 } main158;

static void main125_hclib_async(void *arg, const int ___iter) {
    main125 *ctx = (main125 *)arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int rows; rows = ctx->rows;
    int cols; cols = ctx->cols;
    int size_I; size_I = ctx->size_I;
    int size_R; size_R = ctx->size_R;
    int niter; niter = ctx->niter;
    int iter; iter = ctx->iter;
    int k; k = ctx->k;
    float *I; I = ctx->I;
    float *J; J = ctx->J;
    float q0sqr; q0sqr = ctx->q0sqr;
    float sum; sum = ctx->sum;
    float sum2; sum2 = ctx->sum2;
    float tmp; tmp = ctx->tmp;
    float meanROI; meanROI = ctx->meanROI;
    float varROI; varROI = ctx->varROI;
    float Jc; Jc = ctx->Jc;
    float G2; G2 = ctx->G2;
    float L; L = ctx->L;
    float num; num = ctx->num;
    float den; den = ctx->den;
    float qsqr; qsqr = ctx->qsqr;
    int *iN; iN = ctx->iN;
    int *iS; iS = ctx->iS;
    int *jE; jE = ctx->jE;
    int *jW; jW = ctx->jW;
    float *dN; dN = ctx->dN;
    float *dS; dS = ctx->dS;
    float *dW; dW = ctx->dW;
    float *dE; dE = ctx->dE;
    int r1; r1 = ctx->r1;
    int r2; r2 = ctx->r2;
    int c1; c1 = ctx->c1;
    int c2; c2 = ctx->c2;
    float cN; cN = ctx->cN;
    float cS; cS = ctx->cS;
    float cW; cW = ctx->cW;
    float cE; cE = ctx->cE;
    float *c; c = ctx->c;
    float D; D = ctx->D;
    float lambda; lambda = ctx->lambda;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int nthreads; nthreads = ctx->nthreads;
    hclib_start_finish();
    do {
    int i;     i = ___iter;
{
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
   
		}
  
    }    } while (0);
    hclib_end_finish();
}

static void main158_hclib_async(void *arg, const int ___iter) {
    main158 *ctx = (main158 *)arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int rows; rows = ctx->rows;
    int cols; cols = ctx->cols;
    int size_I; size_I = ctx->size_I;
    int size_R; size_R = ctx->size_R;
    int niter; niter = ctx->niter;
    int iter; iter = ctx->iter;
    int k; k = ctx->k;
    float *I; I = ctx->I;
    float *J; J = ctx->J;
    float q0sqr; q0sqr = ctx->q0sqr;
    float sum; sum = ctx->sum;
    float sum2; sum2 = ctx->sum2;
    float tmp; tmp = ctx->tmp;
    float meanROI; meanROI = ctx->meanROI;
    float varROI; varROI = ctx->varROI;
    float Jc; Jc = ctx->Jc;
    float G2; G2 = ctx->G2;
    float L; L = ctx->L;
    float num; num = ctx->num;
    float den; den = ctx->den;
    float qsqr; qsqr = ctx->qsqr;
    int *iN; iN = ctx->iN;
    int *iS; iS = ctx->iS;
    int *jE; jE = ctx->jE;
    int *jW; jW = ctx->jW;
    float *dN; dN = ctx->dN;
    float *dS; dS = ctx->dS;
    float *dW; dW = ctx->dW;
    float *dE; dE = ctx->dE;
    int r1; r1 = ctx->r1;
    int r2; r2 = ctx->r2;
    int c1; c1 = ctx->c1;
    int c2; c2 = ctx->c2;
    float cN; cN = ctx->cN;
    float cS; cS = ctx->cS;
    float cW; cW = ctx->cW;
    float cE; cE = ctx->cE;
    float *c; c = ctx->c;
    float D; D = ctx->D;
    float lambda; lambda = ctx->lambda;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int nthreads; nthreads = ctx->nthreads;
    hclib_start_finish();
    do {
    int i;     i = ___iter;
{
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
                #ifdef OUTPUT
                //printf("%.5f ", J[k]); 
                #endif //output
            }
	            #ifdef OUTPUT
                //printf("\n"); 
                #endif //output
	     }    } while (0);
    hclib_end_finish();
}

typedef struct _main_entrypoint_ctx {
    int argc;
    char **argv;
    int rows;
    int cols;
    int size_I;
    int size_R;
    int niter;
    int iter;
    int k;
    float *I;
    float *J;
    float q0sqr;
    float sum;
    float sum2;
    float tmp;
    float meanROI;
    float varROI;
    float Jc;
    float G2;
    float L;
    float num;
    float den;
    float qsqr;
    int *iN;
    int *iS;
    int *jE;
    int *jW;
    float *dN;
    float *dS;
    float *dW;
    float *dE;
    int r1;
    int r2;
    int c1;
    int c2;
    float cN;
    float cS;
    float cW;
    float cE;
    float *c;
    float D;
    float lambda;
    int i;
    int j;
    int nthreads;
 } main_entrypoint_ctx;

static void main_entrypoint(void *arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int rows; rows = ctx->rows;
    int cols; cols = ctx->cols;
    int size_I; size_I = ctx->size_I;
    int size_R; size_R = ctx->size_R;
    int niter; niter = ctx->niter;
    int iter; iter = ctx->iter;
    int k; k = ctx->k;
    float *I; I = ctx->I;
    float *J; J = ctx->J;
    float q0sqr; q0sqr = ctx->q0sqr;
    float sum; sum = ctx->sum;
    float sum2; sum2 = ctx->sum2;
    float tmp; tmp = ctx->tmp;
    float meanROI; meanROI = ctx->meanROI;
    float varROI; varROI = ctx->varROI;
    float Jc; Jc = ctx->Jc;
    float G2; G2 = ctx->G2;
    float L; L = ctx->L;
    float num; num = ctx->num;
    float den; den = ctx->den;
    float qsqr; qsqr = ctx->qsqr;
    int *iN; iN = ctx->iN;
    int *iS; iS = ctx->iS;
    int *jE; jE = ctx->jE;
    int *jW; jW = ctx->jW;
    float *dN; dN = ctx->dN;
    float *dS; dS = ctx->dS;
    float *dW; dW = ctx->dW;
    float *dE; dE = ctx->dE;
    int r1; r1 = ctx->r1;
    int r2; r2 = ctx->r2;
    int c1; c1 = ctx->c1;
    int c2; c2 = ctx->c2;
    float cN; cN = ctx->cN;
    float cS; cS = ctx->cS;
    float cW; cW = ctx->cW;
    float cE; cE = ctx->cE;
    float *c; c = ctx->c;
    float D; D = ctx->D;
    float lambda; lambda = ctx->lambda;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int nthreads; nthreads = ctx->nthreads;
for (iter=0; iter< niter; iter++){
		sum=0; sum2=0;     
		for (i=r1; i<=r2; i++) {
            for (j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);
		

		 { 
main125 *ctx = (main125 *)malloc(sizeof(main125));
ctx->argc = argc;
ctx->argv = argv;
ctx->rows = rows;
ctx->cols = cols;
ctx->size_I = size_I;
ctx->size_R = size_R;
ctx->niter = niter;
ctx->iter = iter;
ctx->k = k;
ctx->I = I;
ctx->J = J;
ctx->q0sqr = q0sqr;
ctx->sum = sum;
ctx->sum2 = sum2;
ctx->tmp = tmp;
ctx->meanROI = meanROI;
ctx->varROI = varROI;
ctx->Jc = Jc;
ctx->G2 = G2;
ctx->L = L;
ctx->num = num;
ctx->den = den;
ctx->qsqr = qsqr;
ctx->iN = iN;
ctx->iS = iS;
ctx->jE = jE;
ctx->jW = jW;
ctx->dN = dN;
ctx->dS = dS;
ctx->dW = dW;
ctx->dE = dE;
ctx->r1 = r1;
ctx->r2 = r2;
ctx->c1 = c1;
ctx->c2 = c2;
ctx->cN = cN;
ctx->cS = cS;
ctx->cW = cW;
ctx->cE = cE;
ctx->c = c;
ctx->D = D;
ctx->lambda = lambda;
ctx->i = i;
ctx->j = j;
ctx->nthreads = nthreads;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = rows;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)main125_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
		 { 
main158 *ctx = (main158 *)malloc(sizeof(main158));
ctx->argc = argc;
ctx->argv = argv;
ctx->rows = rows;
ctx->cols = cols;
ctx->size_I = size_I;
ctx->size_R = size_R;
ctx->niter = niter;
ctx->iter = iter;
ctx->k = k;
ctx->I = I;
ctx->J = J;
ctx->q0sqr = q0sqr;
ctx->sum = sum;
ctx->sum2 = sum2;
ctx->tmp = tmp;
ctx->meanROI = meanROI;
ctx->varROI = varROI;
ctx->Jc = Jc;
ctx->G2 = G2;
ctx->L = L;
ctx->num = num;
ctx->den = den;
ctx->qsqr = qsqr;
ctx->iN = iN;
ctx->iS = iS;
ctx->jE = jE;
ctx->jW = jW;
ctx->dN = dN;
ctx->dS = dS;
ctx->dW = dW;
ctx->dE = dE;
ctx->r1 = r1;
ctx->r2 = r2;
ctx->c1 = c1;
ctx->c2 = c2;
ctx->cN = cN;
ctx->cS = cS;
ctx->cW = cW;
ctx->cE = cE;
ctx->c = c;
ctx->D = D;
ctx->lambda = lambda;
ctx->i = i;
ctx->j = j;
ctx->nthreads = nthreads;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = rows;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)main158_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 

	}; }

int main(int argc, char* argv[])
{   
	int rows, cols, size_I, size_R, niter = 10, iter, k;
    float *I, *J, q0sqr, sum, sum2, tmp, meanROI,varROI ;
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW;
	float *dN,*dS,*dW,*dE;
	int r1, r2, c1, c2;
	float cN,cS,cW,cE;
	float *c, D;
	float lambda;
	int i, j;
    int nthreads;

	if (argc == 10)
	{
		rows = atoi(argv[1]); //number of rows in the domain
		cols = atoi(argv[2]); //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
			fprintf(stderr, "rows and cols must be multiples of 16\n");
			exit(1);
		}
		r1   = atoi(argv[3]); //y1 position of the speckle
		r2   = atoi(argv[4]); //y2 position of the speckle
		c1   = atoi(argv[5]); //x1 position of the speckle
		c2   = atoi(argv[6]); //x2 position of the speckle
		nthreads = atoi(argv[7]); // number of threads
		lambda = atof(argv[8]); //Lambda value
		niter = atoi(argv[9]); //number of iterations
	}
    else{
		usage(argc, argv);
    }


	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;
	
	printf("Randomizing the input matrix\n");

    random_matrix(I, rows, cols);

    for (k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
   
	printf("Start the SRAD main loop\n");

#pragma omp_to_hclib body_start
	main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->argc = argc;
ctx->argv = argv;
ctx->rows = rows;
ctx->cols = cols;
ctx->size_I = size_I;
ctx->size_R = size_R;
ctx->niter = niter;
ctx->iter = iter;
ctx->k = k;
ctx->I = I;
ctx->J = J;
ctx->q0sqr = q0sqr;
ctx->sum = sum;
ctx->sum2 = sum2;
ctx->tmp = tmp;
ctx->meanROI = meanROI;
ctx->varROI = varROI;
ctx->Jc = Jc;
ctx->G2 = G2;
ctx->L = L;
ctx->num = num;
ctx->den = den;
ctx->qsqr = qsqr;
ctx->iN = iN;
ctx->iS = iS;
ctx->jE = jE;
ctx->jW = jW;
ctx->dN = dN;
ctx->dS = dS;
ctx->dW = dW;
ctx->dE = dE;
ctx->r1 = r1;
ctx->r2 = r2;
ctx->c1 = c1;
ctx->c2 = c2;
ctx->cN = cN;
ctx->cS = cS;
ctx->cW = cW;
ctx->cE = cE;
ctx->c = c;
ctx->D = D;
ctx->lambda = lambda;
ctx->i = i;
ctx->j = j;
ctx->nthreads = nthreads;
hclib_launch(main_entrypoint, ctx);
free(ctx);

#pragma omp_to_hclib body_end


#ifdef OUTPUT
	  for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){

         printf("%.5f ", J[i * cols + j]); 
    
		}
         printf("\n"); 
   }
#endif 

	printf("Computation Done\n");

	free(I);
	free(J);
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);

	free(c);
	return 0;
}




void random_matrix(float *I, int rows, int cols){

	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		 #ifdef OUTPUT
         //printf("%g ", I[i * cols + j]); 
         #endif 
		}
		 #ifdef OUTPUT
         //printf("\n"); 
         #endif 
	}

}

