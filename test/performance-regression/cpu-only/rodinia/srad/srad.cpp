#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
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

typedef struct _pragma132_omp_parallel {
    int (*rows_ptr);
    int (*cols_ptr);
    int (*size_I_ptr);
    int (*size_R_ptr);
    int (*niter_ptr);
    int (*iter_ptr);
    int k;
    float (*(*I_ptr));
    float (*(*J_ptr));
    float (*q0sqr_ptr);
    float (*sum_ptr);
    float (*sum2_ptr);
    float (*tmp_ptr);
    float (*meanROI_ptr);
    float (*varROI_ptr);
    float Jc;
    float G2;
    float L;
    float num;
    float den;
    float qsqr;
    int (*(*iN_ptr));
    int (*(*iS_ptr));
    int (*(*jE_ptr));
    int (*(*jW_ptr));
    float (*(*dN_ptr));
    float (*(*dS_ptr));
    float (*(*dW_ptr));
    float (*(*dE_ptr));
    int (*r1_ptr);
    int (*r2_ptr);
    int (*c1_ptr);
    int (*c2_ptr);
    float (*cN_ptr);
    float (*cS_ptr);
    float (*cW_ptr);
    float (*cE_ptr);
    float (*(*c_ptr));
    float (*D_ptr);
    float (*lambda_ptr);
    int i;
    int j;
    int (*nthreads_ptr);
    int (*argc_ptr);
    char (*(*(*argv_ptr)));
 } pragma132_omp_parallel;

typedef struct _pragma165_omp_parallel {
    int (*rows_ptr);
    int (*cols_ptr);
    int (*size_I_ptr);
    int (*size_R_ptr);
    int (*niter_ptr);
    int (*iter_ptr);
    int k;
    float (*(*I_ptr));
    float (*(*J_ptr));
    float (*q0sqr_ptr);
    float (*sum_ptr);
    float (*sum2_ptr);
    float (*tmp_ptr);
    float (*meanROI_ptr);
    float (*varROI_ptr);
    float (*Jc_ptr);
    float (*G2_ptr);
    float (*L_ptr);
    float (*num_ptr);
    float (*den_ptr);
    float (*qsqr_ptr);
    int (*(*iN_ptr));
    int (*(*iS_ptr));
    int (*(*jE_ptr));
    int (*(*jW_ptr));
    float (*(*dN_ptr));
    float (*(*dS_ptr));
    float (*(*dW_ptr));
    float (*(*dE_ptr));
    int (*r1_ptr);
    int (*r2_ptr);
    int (*c1_ptr);
    int (*c2_ptr);
    float cN;
    float cS;
    float cW;
    float cE;
    float (*(*c_ptr));
    float D;
    float (*lambda_ptr);
    int i;
    int j;
    int (*nthreads_ptr);
    int (*argc_ptr);
    char (*(*(*argv_ptr)));
 } pragma165_omp_parallel;

static void pragma132_omp_parallel_hclib_async(void *____arg, const int ___iter0);
static void pragma165_omp_parallel_hclib_async(void *____arg, const int ___iter0);
typedef struct _main_entrypoint_ctx {
    int rows;
    int cols;
    int size_I;
    int size_R;
    int niter;
    int iter;
    int k;
    float (*I);
    float (*J);
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
    int (*iN);
    int (*iS);
    int (*jE);
    int (*jW);
    float (*dN);
    float (*dS);
    float (*dW);
    float (*dE);
    int r1;
    int r2;
    int c1;
    int c2;
    float cN;
    float cS;
    float cW;
    float cE;
    float (*c);
    float D;
    float lambda;
    int i;
    int j;
    int nthreads;
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int rows; rows = ctx->rows;
    int cols; cols = ctx->cols;
    int size_I; size_I = ctx->size_I;
    int size_R; size_R = ctx->size_R;
    int niter; niter = ctx->niter;
    int iter; iter = ctx->iter;
    int k; k = ctx->k;
    float (*I); I = ctx->I;
    float (*J); J = ctx->J;
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
    int (*iN); iN = ctx->iN;
    int (*iS); iS = ctx->iS;
    int (*jE); jE = ctx->jE;
    int (*jW); jW = ctx->jW;
    float (*dN); dN = ctx->dN;
    float (*dS); dS = ctx->dS;
    float (*dW); dW = ctx->dW;
    float (*dE); dE = ctx->dE;
    int r1; r1 = ctx->r1;
    int r2; r2 = ctx->r2;
    int c1; c1 = ctx->c1;
    int c2; c2 = ctx->c2;
    float cN; cN = ctx->cN;
    float cS; cS = ctx->cS;
    float cW; cW = ctx->cW;
    float cE; cE = ctx->cE;
    float (*c); c = ctx->c;
    float D; D = ctx->D;
    float lambda; lambda = ctx->lambda;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int nthreads; nthreads = ctx->nthreads;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
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
pragma132_omp_parallel *new_ctx = (pragma132_omp_parallel *)malloc(sizeof(pragma132_omp_parallel));
new_ctx->rows_ptr = &(rows);
new_ctx->cols_ptr = &(cols);
new_ctx->size_I_ptr = &(size_I);
new_ctx->size_R_ptr = &(size_R);
new_ctx->niter_ptr = &(niter);
new_ctx->iter_ptr = &(iter);
new_ctx->k = k;
new_ctx->I_ptr = &(I);
new_ctx->J_ptr = &(J);
new_ctx->q0sqr_ptr = &(q0sqr);
new_ctx->sum_ptr = &(sum);
new_ctx->sum2_ptr = &(sum2);
new_ctx->tmp_ptr = &(tmp);
new_ctx->meanROI_ptr = &(meanROI);
new_ctx->varROI_ptr = &(varROI);
new_ctx->Jc = Jc;
new_ctx->G2 = G2;
new_ctx->L = L;
new_ctx->num = num;
new_ctx->den = den;
new_ctx->qsqr = qsqr;
new_ctx->iN_ptr = &(iN);
new_ctx->iS_ptr = &(iS);
new_ctx->jE_ptr = &(jE);
new_ctx->jW_ptr = &(jW);
new_ctx->dN_ptr = &(dN);
new_ctx->dS_ptr = &(dS);
new_ctx->dW_ptr = &(dW);
new_ctx->dE_ptr = &(dE);
new_ctx->r1_ptr = &(r1);
new_ctx->r2_ptr = &(r2);
new_ctx->c1_ptr = &(c1);
new_ctx->c2_ptr = &(c2);
new_ctx->cN_ptr = &(cN);
new_ctx->cS_ptr = &(cS);
new_ctx->cW_ptr = &(cW);
new_ctx->cE_ptr = &(cE);
new_ctx->c_ptr = &(c);
new_ctx->D_ptr = &(D);
new_ctx->lambda_ptr = &(lambda);
new_ctx->i = i;
new_ctx->j = j;
new_ctx->nthreads_ptr = &(nthreads);
new_ctx->argc_ptr = &(argc);
new_ctx->argv_ptr = &(argv);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = rows;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma132_omp_parallel_hclib_async, new_ctx, NULL, 1, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
 } 
 { 
pragma165_omp_parallel *new_ctx = (pragma165_omp_parallel *)malloc(sizeof(pragma165_omp_parallel));
new_ctx->rows_ptr = &(rows);
new_ctx->cols_ptr = &(cols);
new_ctx->size_I_ptr = &(size_I);
new_ctx->size_R_ptr = &(size_R);
new_ctx->niter_ptr = &(niter);
new_ctx->iter_ptr = &(iter);
new_ctx->k = k;
new_ctx->I_ptr = &(I);
new_ctx->J_ptr = &(J);
new_ctx->q0sqr_ptr = &(q0sqr);
new_ctx->sum_ptr = &(sum);
new_ctx->sum2_ptr = &(sum2);
new_ctx->tmp_ptr = &(tmp);
new_ctx->meanROI_ptr = &(meanROI);
new_ctx->varROI_ptr = &(varROI);
new_ctx->Jc_ptr = &(Jc);
new_ctx->G2_ptr = &(G2);
new_ctx->L_ptr = &(L);
new_ctx->num_ptr = &(num);
new_ctx->den_ptr = &(den);
new_ctx->qsqr_ptr = &(qsqr);
new_ctx->iN_ptr = &(iN);
new_ctx->iS_ptr = &(iS);
new_ctx->jE_ptr = &(jE);
new_ctx->jW_ptr = &(jW);
new_ctx->dN_ptr = &(dN);
new_ctx->dS_ptr = &(dS);
new_ctx->dW_ptr = &(dW);
new_ctx->dE_ptr = &(dE);
new_ctx->r1_ptr = &(r1);
new_ctx->r2_ptr = &(r2);
new_ctx->c1_ptr = &(c1);
new_ctx->c2_ptr = &(c2);
new_ctx->cN = cN;
new_ctx->cS = cS;
new_ctx->cW = cW;
new_ctx->cE = cE;
new_ctx->c_ptr = &(c);
new_ctx->D = D;
new_ctx->lambda_ptr = &(lambda);
new_ctx->i = i;
new_ctx->j = j;
new_ctx->nthreads_ptr = &(nthreads);
new_ctx->argc_ptr = &(argc);
new_ctx->argv_ptr = &(argv);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = rows;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma165_omp_parallel_hclib_async, new_ctx, NULL, 1, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
 } 

	} ;     free(____arg);
}

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

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->rows = rows;
new_ctx->cols = cols;
new_ctx->size_I = size_I;
new_ctx->size_R = size_R;
new_ctx->niter = niter;
new_ctx->iter = iter;
new_ctx->k = k;
new_ctx->I = I;
new_ctx->J = J;
new_ctx->q0sqr = q0sqr;
new_ctx->sum = sum;
new_ctx->sum2 = sum2;
new_ctx->tmp = tmp;
new_ctx->meanROI = meanROI;
new_ctx->varROI = varROI;
new_ctx->Jc = Jc;
new_ctx->G2 = G2;
new_ctx->L = L;
new_ctx->num = num;
new_ctx->den = den;
new_ctx->qsqr = qsqr;
new_ctx->iN = iN;
new_ctx->iS = iS;
new_ctx->jE = jE;
new_ctx->jW = jW;
new_ctx->dN = dN;
new_ctx->dS = dS;
new_ctx->dW = dW;
new_ctx->dE = dE;
new_ctx->r1 = r1;
new_ctx->r2 = r2;
new_ctx->c1 = c1;
new_ctx->c2 = c2;
new_ctx->cN = cN;
new_ctx->cS = cS;
new_ctx->cW = cW;
new_ctx->cE = cE;
new_ctx->c = c;
new_ctx->D = D;
new_ctx->lambda = lambda;
new_ctx->i = i;
new_ctx->j = j;
new_ctx->nthreads = nthreads;
new_ctx->argc = argc;
new_ctx->argv = argv;
hclib_launch(main_entrypoint, new_ctx);



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
static void pragma132_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma132_omp_parallel *ctx = (pragma132_omp_parallel *)____arg;
    int k; k = ctx->k;
    float Jc; Jc = ctx->Jc;
    float G2; G2 = ctx->G2;
    float L; L = ctx->L;
    float num; num = ctx->num;
    float den; den = ctx->den;
    float qsqr; qsqr = ctx->qsqr;
    int i; i = ctx->i;
    int j; j = ctx->j;
    do {
    int i;     i = ___iter0;
{
            for (int j = 0; j < (*(ctx->cols_ptr)); j++) { 
		
				k = i * (*(ctx->cols_ptr)) + j;
				Jc = (*(ctx->J_ptr))[k];
 
				// directional derivates
                (*(ctx->dN_ptr))[k] = (*(ctx->J_ptr))[(*(ctx->iN_ptr))[i] * (*(ctx->cols_ptr)) + j] - Jc;
                (*(ctx->dS_ptr))[k] = (*(ctx->J_ptr))[(*(ctx->iS_ptr))[i] * (*(ctx->cols_ptr)) + j] - Jc;
                (*(ctx->dW_ptr))[k] = (*(ctx->J_ptr))[i * (*(ctx->cols_ptr)) + (*(ctx->jW_ptr))[j]] - Jc;
                (*(ctx->dE_ptr))[k] = (*(ctx->J_ptr))[i * (*(ctx->cols_ptr)) + (*(ctx->jE_ptr))[j]] - Jc;
			
                G2 = ((*(ctx->dN_ptr))[k]*(*(ctx->dN_ptr))[k] + (*(ctx->dS_ptr))[k]*(*(ctx->dS_ptr))[k] 
                    + (*(ctx->dW_ptr))[k]*(*(ctx->dW_ptr))[k] + (*(ctx->dE_ptr))[k]*(*(ctx->dE_ptr))[k]) / (Jc*Jc);

   		        L = ((*(ctx->dN_ptr))[k] + (*(ctx->dS_ptr))[k] + (*(ctx->dW_ptr))[k] + (*(ctx->dE_ptr))[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-(*(ctx->q0sqr_ptr))) / ((*(ctx->q0sqr_ptr)) * (1+(*(ctx->q0sqr_ptr)))) ;
                (*(ctx->c_ptr))[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if ((*(ctx->c_ptr))[k] < 0) {(*(ctx->c_ptr))[k] = 0;}
                else if ((*(ctx->c_ptr))[k] > 1) {(*(ctx->c_ptr))[k] = 1;}
   
		}
  
    } ;     } while (0);
}


static void pragma165_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma165_omp_parallel *ctx = (pragma165_omp_parallel *)____arg;
    int k; k = ctx->k;
    float cN; cN = ctx->cN;
    float cS; cS = ctx->cS;
    float cW; cW = ctx->cW;
    float cE; cE = ctx->cE;
    float D; D = ctx->D;
    int i; i = ctx->i;
    int j; j = ctx->j;
    do {
    int i;     i = ___iter0;
{
            for (int j = 0; j < (*(ctx->cols_ptr)); j++) {        

                // current index
                k = i * (*(ctx->cols_ptr)) + j;
                
                // diffusion coefficent
					cN = (*(ctx->c_ptr))[k];
					cS = (*(ctx->c_ptr))[(*(ctx->iS_ptr))[i] * (*(ctx->cols_ptr)) + j];
					cW = (*(ctx->c_ptr))[k];
					cE = (*(ctx->c_ptr))[i * (*(ctx->cols_ptr)) + (*(ctx->jE_ptr))[j]];

                // divergence (equ 58)
                D = cN * (*(ctx->dN_ptr))[k] + cS * (*(ctx->dS_ptr))[k] + cW * (*(ctx->dW_ptr))[k] + cE * (*(ctx->dE_ptr))[k];
                
                // image update (equ 61)
                (*(ctx->J_ptr))[k] = (*(ctx->J_ptr))[k] + 0.25*(*(ctx->lambda_ptr))*D;
                #ifdef OUTPUT
                //printf("%.5f ", J[k]); 
                #endif //output
            }
	            #ifdef OUTPUT
                //printf("\n"); 
                #endif //output
	     } ;     } while (0);
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

