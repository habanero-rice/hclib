#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef FP_NUMBER
typedef double FP_NUMBER;
#else
typedef float FP_NUMBER;
#endif


#define GET_RAND_FP ((FP_NUMBER)rand()/((FP_NUMBER)(RAND_MAX)+(FP_NUMBER)(1)))
char L_FNAME[32], U_FNAME[32], A_FNAME[32];

typedef struct _pragma76_omp_parallel {
    int i;
    int j;
    int (*MatrixDim_ptr);
    float (*(*(*L_ptr)));
    float (*(*(*U_ptr)));
 } pragma76_omp_parallel;

typedef struct _pragma92_omp_parallel {
    int i;
    int j;
    int k;
    int (*MatrixDim_ptr);
    float sum;
    float (*(*(*L_ptr)));
    float (*(*(*U_ptr)));
    float (*(*(*A_ptr)));
 } pragma92_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma76_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int idx) {
        }
};

#else
static void pragma76_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif

#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma92_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int idx) {
        }
};

#else
static void pragma92_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
int main (int argc, char **argv){
    int i,j,k,MatrixDim;
    FP_NUMBER sum, **L, **U, **A;
    FILE *fl,*fu,*fa;

    if ( argc < 2) {
        printf("./gen_input [Matrix_Dimension_size]\n");
        return 1;
    }

    MatrixDim = atoi(argv[1]);
    L = (FP_NUMBER **) malloc(sizeof(FP_NUMBER*)*MatrixDim);
    U = (FP_NUMBER **) malloc(sizeof(FP_NUMBER*)*MatrixDim);
    A = (FP_NUMBER **) malloc(sizeof(FP_NUMBER*)*MatrixDim);

    if ( !L || !U || !A){
        printf("Can not allocate memory\n");
        if (L) free(L);
        if (U) free(U);
        if (A) free(A);
        return 1;
    }

    srand(time(NULL));

    sprintf(L_FNAME, "l-%d.dat", MatrixDim);
    fl = fopen(L_FNAME, "wb");
    if (fl == NULL) {
        printf("Cannot open file %s\n", L_FNAME);
        return 1;
    }

    sprintf(U_FNAME, "u-%d.dat", MatrixDim);
    fu = fopen(U_FNAME, "wb");
    if (fu == NULL) {
        printf("Cannot open file %s\n", U_FNAME);
        return 1;
    }

    sprintf(A_FNAME, "%d.dat", MatrixDim);
    fa = fopen(A_FNAME, "wb");
    if (!fa) {
        printf("Cannot open file %s\n", A_FNAME);
        return 1;
    }

    for (i=0; i < MatrixDim; i ++){
        L[i]=(FP_NUMBER*)malloc(sizeof(FP_NUMBER)*MatrixDim);
        U[i]=(FP_NUMBER*)malloc(sizeof(FP_NUMBER)*MatrixDim);
        A[i]=(FP_NUMBER*)malloc(sizeof(FP_NUMBER)*MatrixDim);
    }
 { 
pragma76_omp_parallel *new_ctx = (pragma76_omp_parallel *)malloc(sizeof(pragma76_omp_parallel));
new_ctx->i = i;
new_ctx->j = j;
new_ctx->MatrixDim_ptr = &(MatrixDim);
new_ctx->L_ptr = &(L);
new_ctx->U_ptr = &(U);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = MatrixDim;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((MatrixDim) - (0), pragma76_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma76_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 

 { 
pragma92_omp_parallel *new_ctx = (pragma92_omp_parallel *)malloc(sizeof(pragma92_omp_parallel));
new_ctx->i = i;
new_ctx->j = j;
new_ctx->k = k;
new_ctx->MatrixDim_ptr = &(MatrixDim);
new_ctx->sum = sum;
new_ctx->L_ptr = &(L);
new_ctx->U_ptr = &(U);
new_ctx->A_ptr = &(A);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = MatrixDim;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((MatrixDim) - (0), pragma92_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma92_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 

    for (i=0; i < MatrixDim; i ++) {
        for (j=0; j < MatrixDim; j++)
            fprintf(fl, "%f ", L[i][j]);
        fprintf(fl, "\n");
    }
    fclose(fl);

    for (i=0; i < MatrixDim; i ++) {
        for (j=0; j < MatrixDim; j++)
            fprintf(fu, "%f ", U[i][j]);
        fprintf(fu, "\n");
    }
    fclose(fu);

    fprintf(fa, "%d\n", MatrixDim);
    for (i=0; i < MatrixDim; i ++) {
        for (j=0; j < MatrixDim; j++)
            fprintf(fa, "%f ", A[i][j]);
        fprintf(fa, "\n");
    }
    fclose(fa);

    for (i = 0; i < MatrixDim; i ++ ){
        free(L[i]);
        free(U[i]);
        free(A[i]);
    }
    free(L);
    free(U);
    free(A);

    return 0;
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma76_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma76_omp_parallel *ctx = (pragma76_omp_parallel *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    hclib_start_finish();
    do {
    i = ___iter0;
{
        for (j=0; j < (*(ctx->MatrixDim_ptr)); j++){
            if ( i == j) {
                (*(ctx->L_ptr))[i][j] = 1.0;
                (*(ctx->U_ptr))[i][j] = GET_RAND_FP;
            } else if (i < j){
                (*(ctx->L_ptr))[i][j] = 0;
                (*(ctx->U_ptr))[i][j] = GET_RAND_FP;
            } else { // i > j
                (*(ctx->L_ptr))[i][j] = GET_RAND_FP;
                (*(ctx->U_ptr))[i][j] = 0;
            }
        }
    } ;     } while (0);
    ; hclib_end_finish_nonblocking();

}

#endif


#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma92_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma92_omp_parallel *ctx = (pragma92_omp_parallel *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int k; k = ctx->k;
    float sum; sum = ctx->sum;
    do {
    i = ___iter0;
{
        for (j=0; j < (*(ctx->MatrixDim_ptr)); j++){
            sum = 0;
            for(k=0; k < (*(ctx->MatrixDim_ptr)); k++)
                sum += (*(ctx->L_ptr))[i][k]*(*(ctx->U_ptr))[k][j];
            (*(ctx->A_ptr))[i][j] = sum;
        }
    } ;     } while (0);
}

#endif

