#include "hclib.h"
#include <stdio.h>
#include <omp.h>

extern int omp_num_threads;

#define BS 16

#define AA(_i,_j) a[offset*size+_i*size+_j+offset]
#define BB(_i,_j) a[_i*size+_j]

void lud_diagonal_omp (float* a, int size, int offset)
{
    int i, j, k;
    for (i = 0; i < BS; i++) {

        for (j = i; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(i,j) = AA(i,j) - AA(i,k) * AA(k,j);
            }
        }
   
        float temp = 1.f/AA(i,i);
        for (j = i+1; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(j,i) = AA(j,i) - AA(j,k) * AA(k,i);
            }
            AA(j,i) = AA(j,i)*temp;
        }
    }

}

// implements block LU factorization 
typedef struct _lud_omp49 {
    float *a;
    int size;
    int offset;
    int chunk_idx;
    int size_inter;
    int chunks_in_inter_row;
    int chunks_per_inter;
 } lud_omp49;

typedef struct _lud_omp103 {
    float *a;
    int size;
    int offset;
    int chunk_idx;
    int size_inter;
    int chunks_in_inter_row;
    int chunks_per_inter;
 } lud_omp103;

static void lud_omp49_hclib_async(void *arg, const int ___iter) {
    lud_omp49 *ctx = (lud_omp49 *)arg;
    float *a; a = ctx->a;
    int size; size = ctx->size;
    int offset; offset = ctx->offset;
    int chunk_idx; chunk_idx = ctx->chunk_idx;
    int size_inter; size_inter = ctx->size_inter;
    int chunks_in_inter_row; chunks_in_inter_row = ctx->chunks_in_inter_row;
    int chunks_per_inter; chunks_per_inter = ctx->chunks_per_inter;
    chunk_idx = ___iter;
    do {
{
            int i, j, k, i_global, j_global, i_here, j_here;
            float sum;           
            float temp[BS*BS] __attribute__ ((aligned (64)));

            for (i = 0; i < BS; i++) {
                for (j =0; j < BS; j++){
                    temp[i*BS + j] = a[size*(i + offset) + offset + j ];
                }
            }
            i_global = offset;
            j_global = offset;
            
            // processing top perimeter
            //
            j_global += BS * (chunk_idx+1);
            for (j = 0; j < BS; j++) {
                for (i = 0; i < BS; i++) {
                    sum = 0.f;
                    for (k=0; k < i; k++) {
                        sum += temp[BS*i +k] * BB((i_global+k),(j_global+j));
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    BB(i_here, j_here) = BB(i_here,j_here) - sum;
                }
            }

            // processing left perimeter
            //
            j_global = offset;
            i_global += BS * (chunk_idx + 1);
            for (i = 0; i < BS; i++) {
                for (j = 0; j < BS; j++) {
                    sum = 0.f;
                    for (k=0; k < j; k++) {
                        sum += BB((i_global+i),(j_global+k)) * temp[BS*k + j];
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    a[size*i_here + j_here] = ( a[size*i_here+j_here] - sum ) / a[size*(offset+j) + offset+j];
                }
            }

        }    } while (0);
}

static void lud_omp103_hclib_async(void *arg, const int ___iter) {
    lud_omp103 *ctx = (lud_omp103 *)arg;
    float *a; a = ctx->a;
    int size; size = ctx->size;
    int offset; offset = ctx->offset;
    int chunk_idx; chunk_idx = ctx->chunk_idx;
    int size_inter; size_inter = ctx->size_inter;
    int chunks_in_inter_row; chunks_in_inter_row = ctx->chunks_in_inter_row;
    int chunks_per_inter; chunks_per_inter = ctx->chunks_per_inter;
    chunk_idx = ___iter;
    do {
{
            int i, j, k, i_global, j_global;
            float temp_top[BS*BS] __attribute__ ((aligned (64)));
            float temp_left[BS*BS] __attribute__ ((aligned (64)));
            float sum[BS] __attribute__ ((aligned (64))) = {0.f};
            
            i_global = offset + BS * (1 +  chunk_idx/chunks_in_inter_row);
            j_global = offset + BS * (1 + chunk_idx%chunks_in_inter_row);

            for (i = 0; i < BS; i++) {
                for (j =0; j < BS; j++){
                    temp_top[i*BS + j]  = a[size*(i + offset) + j + j_global ];
                    temp_left[i*BS + j] = a[size*(i + i_global) + offset + j];
                }
            }

            for (i = 0; i < BS; i++)
            {
                for (k=0; k < BS; k++) {
                    for (j = 0; j < BS; j++) {
                        sum[j] += temp_left[BS*i + k] * temp_top[BS*k + j];
                    }
                }
                for (j = 0; j < BS; j++) {
                    BB((i+i_global),(j+j_global)) -= sum[j];
                    sum[j] = 0.f;
                }
            }
        }    } while (0);
}

void lud_omp(float *a, int size)
{
    int offset, chunk_idx, size_inter, chunks_in_inter_row, chunks_per_inter;

    for (offset = 0; offset < size - BS ; offset += BS)
    {
        // lu factorization of left-top corner block diagonal matrix 
        //
        lud_diagonal_omp(a, size, offset);
            
        size_inter = size - offset -  BS;
        chunks_in_inter_row  = size_inter/BS;
        
        // calculate perimeter block matrices
        // 
         { 
lud_omp49 *ctx = (lud_omp49 *)malloc(sizeof(lud_omp49));
ctx->a = a;
ctx->size = size;
ctx->offset = offset;
ctx->chunk_idx = chunk_idx;
ctx->size_inter = size_inter;
ctx->chunks_in_inter_row = chunks_in_inter_row;
ctx->chunks_per_inter = chunks_per_inter;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = chunks_in_inter_row;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)lud_omp49_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
        
        // update interior block matrices
        //
        chunks_per_inter = chunks_in_inter_row*chunks_in_inter_row;

         { 
lud_omp103 *ctx = (lud_omp103 *)malloc(sizeof(lud_omp103));
ctx->a = a;
ctx->size = size;
ctx->offset = offset;
ctx->chunk_idx = chunk_idx;
ctx->size_inter = size_inter;
ctx->chunks_in_inter_row = chunks_in_inter_row;
ctx->chunks_per_inter = chunks_per_inter;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = chunks_per_inter;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)lud_omp103_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
    }

    lud_diagonal_omp(a, size, offset);
}
