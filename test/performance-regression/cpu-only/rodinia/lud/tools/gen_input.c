#include "hclib.h"
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

typedef struct _pragma67 {
    int argc;
    char **argv;
    int i;
    int j;
    int k;
    int MatrixDim;
    FP_NUMBER sum;
    FP_NUMBER **L;
    FP_NUMBER **U;
    FP_NUMBER **A;
    FILE *fl;
    FILE *fu;
    FILE *fa;
 } pragma67;

typedef struct _pragma83 {
    int argc;
    char **argv;
    int i;
    int j;
    int k;
    int MatrixDim;
    FP_NUMBER sum;
    FP_NUMBER **L;
    FP_NUMBER **U;
    FP_NUMBER **A;
    FILE *fl;
    FILE *fu;
    FILE *fa;
 } pragma83;

static void pragma67_hclib_async(void *____arg, const int ___iter);
static void pragma83_hclib_async(void *____arg, const int ___iter);
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
pragma67 *ctx = (pragma67 *)malloc(sizeof(pragma67));
ctx->argc = argc;
ctx->argv = argv;
ctx->i = i;
ctx->j = j;
ctx->k = k;
ctx->MatrixDim = MatrixDim;
ctx->sum = sum;
ctx->L = L;
ctx->U = U;
ctx->A = A;
ctx->fl = fl;
ctx->fu = fu;
ctx->fa = fa;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = MatrixDim;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma67_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 

 { 
pragma83 *ctx = (pragma83 *)malloc(sizeof(pragma83));
ctx->argc = argc;
ctx->argv = argv;
ctx->i = i;
ctx->j = j;
ctx->k = k;
ctx->MatrixDim = MatrixDim;
ctx->sum = sum;
ctx->L = L;
ctx->U = U;
ctx->A = A;
ctx->fl = fl;
ctx->fu = fu;
ctx->fa = fa;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = MatrixDim;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma83_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
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
} static void pragma67_hclib_async(void *____arg, const int ___iter) {
    pragma67 *ctx = (pragma67 *)____arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int k; k = ctx->k;
    int MatrixDim; MatrixDim = ctx->MatrixDim;
    FP_NUMBER sum; sum = ctx->sum;
    FP_NUMBER **L; L = ctx->L;
    FP_NUMBER **U; U = ctx->U;
    FP_NUMBER **A; A = ctx->A;
    FILE *fl; fl = ctx->fl;
    FILE *fu; fu = ctx->fu;
    FILE *fa; fa = ctx->fa;
    hclib_start_finish();
    do {
    i = ___iter;
{
        for (j=0; j < MatrixDim; j++){
            if ( i == j) {
                L[i][j] = 1.0;
                U[i][j] = GET_RAND_FP;
            } else if (i < j){
                L[i][j] = 0;
                U[i][j] = GET_RAND_FP;
            } else { // i > j
                L[i][j] = GET_RAND_FP;
                U[i][j] = 0;
            }
        }
    } ;     } while (0);
    ; hclib_end_finish();
}

static void pragma83_hclib_async(void *____arg, const int ___iter) {
    pragma83 *ctx = (pragma83 *)____arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int k; k = ctx->k;
    int MatrixDim; MatrixDim = ctx->MatrixDim;
    FP_NUMBER sum; sum = ctx->sum;
    FP_NUMBER **L; L = ctx->L;
    FP_NUMBER **U; U = ctx->U;
    FP_NUMBER **A; A = ctx->A;
    FILE *fl; fl = ctx->fl;
    FILE *fu; fu = ctx->fu;
    FILE *fa; fa = ctx->fa;
    hclib_start_finish();
    do {
    i = ___iter;
{
        for (j=0; j < MatrixDim; j++){
            sum = 0;
            for(k=0; k < MatrixDim; k++)
                sum += L[i][k]*U[k][j];
            A[i][j] = sum;
        }
    } ;     } while (0);
    ; hclib_end_finish();
}


