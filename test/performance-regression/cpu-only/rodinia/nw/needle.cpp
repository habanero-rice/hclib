#include "hclib.h"
#define LIMIT -999
//#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// Returns the current system time in microseconds 
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;

}

int maximum( int a,
		 int b,
		 int c){

	int k;
	if( a <= b )
		k = b;
	else 
	k = a;

	if( k <=c )
	return(c);
	else
	return(k);
}


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	fprintf(stderr, "\t<num_threads>    - no. of threads\n");
	exit(1);
}

typedef struct _nw_optimized100 {
    int *input_itemsets;
    int *output_itemsets;
    int *referrence;
    int max_rows;
    int max_cols;
    int penalty;
    int blk;
 } nw_optimized100;

typedef struct _nw_optimized155 {
    int *input_itemsets;
    int *output_itemsets;
    int *referrence;
    int max_rows;
    int max_cols;
    int penalty;
    int blk;
 } nw_optimized155;

static void nw_optimized100_hclib_async(void *arg, const int ___iter) {
    nw_optimized100 *ctx = (nw_optimized100 *)arg;
    int *input_itemsets; input_itemsets = ctx->input_itemsets;
    int *output_itemsets; output_itemsets = ctx->output_itemsets;
    int *referrence; referrence = ctx->referrence;
    int max_rows; max_rows = ctx->max_rows;
    int max_cols; max_cols = ctx->max_cols;
    int penalty; penalty = ctx->penalty;
    int blk; blk = ctx->blk;
    hclib_start_finish();
    do {
    int b_index_x;     b_index_x = ___iter;
{
            int b_index_y = blk - 1 - b_index_x;
            int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)] __attribute__ ((aligned (64)));
            int reference_l[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

            // Copy referrence to local memory
            for ( int i = 0; i < BLOCK_SIZE; ++i )
            {
                for ( int j = 0; j < BLOCK_SIZE; ++j)
                {
                    reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
                }
            }

            // Copy input_itemsets to local memory
            for ( int i = 0; i < BLOCK_SIZE + 1; ++i )
            {
                for ( int j = 0; j < BLOCK_SIZE + 1; ++j)
                {
                    input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
                }
            }

            // Compute
            for ( int i = 1; i < BLOCK_SIZE + 1; ++i )
            {
                for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
                {
                    input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
                            input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
                            input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
                }
            }

            // Copy results to global memory
            for ( int i = 0; i < BLOCK_SIZE; ++i )
            {
                for ( int j = 0; j < BLOCK_SIZE; ++j)
                {
                    input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j + 1];
                }
            }
            
        }    } while (0);
    hclib_end_finish();
}

static void nw_optimized155_hclib_async(void *arg, const int ___iter) {
    nw_optimized155 *ctx = (nw_optimized155 *)arg;
    int *input_itemsets; input_itemsets = ctx->input_itemsets;
    int *output_itemsets; output_itemsets = ctx->output_itemsets;
    int *referrence; referrence = ctx->referrence;
    int max_rows; max_rows = ctx->max_rows;
    int max_cols; max_cols = ctx->max_cols;
    int penalty; penalty = ctx->penalty;
    int blk; blk = ctx->blk;
    hclib_start_finish();
    do {
    int b_index_x;     b_index_x = ___iter;
{
            int b_index_y = (max_cols-1)/BLOCK_SIZE + blk - 2 - b_index_x;

            int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)] __attribute__ ((aligned (64)));
            int reference_l[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));
 
            // Copy referrence to local memory
            for ( int i = 0; i < BLOCK_SIZE; ++i )
            {
                for ( int j = 0; j < BLOCK_SIZE; ++j)
                {
                    reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
                }
            }

            // Copy input_itemsets to local memory
            for ( int i = 0; i < BLOCK_SIZE + 1; ++i )
            {
                for ( int j = 0; j < BLOCK_SIZE + 1; ++j)
                {
                    input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
                }
            }

            // Compute
            for ( int i = 1; i < BLOCK_SIZE + 1; ++i )
            {
                for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
                {
                    input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
                            input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
                            input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
                }
            }

            // Copy results to global memory
            for ( int i = 0; i < BLOCK_SIZE; ++i )
            {
                for ( int j = 0; j < BLOCK_SIZE; ++j)
                {
                    input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j +1];
                }
            }
        }    } while (0);
    hclib_end_finish();
}

void nw_optimized(int *input_itemsets, int *output_itemsets, int *referrence,
        int max_rows, int max_cols, int penalty)
{
    for( int blk = 1; blk <= (max_cols-1)/BLOCK_SIZE; blk++ )
    {
         { 
nw_optimized100 *ctx = (nw_optimized100 *)malloc(sizeof(nw_optimized100));
ctx->input_itemsets = input_itemsets;
ctx->output_itemsets = output_itemsets;
ctx->referrence = referrence;
ctx->max_rows = max_rows;
ctx->max_cols = max_cols;
ctx->penalty = penalty;
ctx->blk = blk;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = blk;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)nw_optimized100_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
    }    
        
    printf("Processing bottom-right matrix\n");

    for ( int blk = 2; blk <= (max_cols-1)/BLOCK_SIZE; blk++ )
    {
         { 
nw_optimized155 *ctx = (nw_optimized155 *)malloc(sizeof(nw_optimized155));
ctx->input_itemsets = input_itemsets;
ctx->output_itemsets = output_itemsets;
ctx->referrence = referrence;
ctx->max_rows = max_rows;
ctx->max_cols = max_cols;
ctx->penalty = penalty;
ctx->blk = blk;
hclib_loop_domain_t domain;
domain.low = blk - 1;
domain.high = (max_cols - 1) / 16;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)nw_optimized155_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
    }

}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
typedef struct _main_entrypoint_ctx {
    int argc;
    char **argv;
    int max_rows;
    int max_cols;
    int penalty;
    int *input_itemsets;
    int *output_itemsets;
    int *referrence;
    int omp_num_threads;
    long long start_time;
 } main_entrypoint_ctx;

static void main_entrypoint(void *arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int max_rows; max_rows = ctx->max_rows;
    int max_cols; max_cols = ctx->max_cols;
    int penalty; penalty = ctx->penalty;
    int *input_itemsets; input_itemsets = ctx->input_itemsets;
    int *output_itemsets; output_itemsets = ctx->output_itemsets;
    int *referrence; referrence = ctx->referrence;
    int omp_num_threads; omp_num_threads = ctx->omp_num_threads;
    long long start_time; start_time = ctx->start_time;
nw_optimized( input_itemsets, output_itemsets, referrence,
        max_rows, max_cols, penalty ); }

void
runTest( int argc, char** argv) 
{
    int max_rows, max_cols, penalty;
    int *input_itemsets, *output_itemsets, *referrence;
    //int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
    //int size;
    int omp_num_threads;


    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_cols
    if (argc == 4)
    {
        max_rows = atoi(argv[1]);
        max_cols = atoi(argv[1]);
        penalty = atoi(argv[2]);
        omp_num_threads = atoi(argv[3]);
    }
    else{
        usage(argc, argv);
    }

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
    output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );


    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );

    for (int i = 0 ; i < max_cols; i++){
        for (int j = 0 ; j < max_rows; j++){
            input_itemsets[i*max_cols+j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

    for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
        input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
        input_itemsets[j] = rand() % 10 + 1;
    }


    for (int i = 1 ; i < max_cols; i++){
        for (int j = 1 ; j < max_rows; j++){
            referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for( int i = 1; i< max_rows ; i++)
        input_itemsets[i*max_cols] = -i * penalty;
    for( int j = 1; j< max_cols ; j++)
        input_itemsets[j] = -j * penalty;

    //Compute top-left matrix 
    printf("Num of threads: %d\n", omp_num_threads);
    printf("Processing top-left matrix\n");
   
    long long start_time = get_time();
#pragma omp_to_hclib body_start

    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->argc = argc;
ctx->argv = argv;
ctx->max_rows = max_rows;
ctx->max_cols = max_cols;
ctx->penalty = penalty;
ctx->input_itemsets = input_itemsets;
ctx->output_itemsets = output_itemsets;
ctx->referrence = referrence;
ctx->omp_num_threads = omp_num_threads;
ctx->start_time = start_time;
hclib_launch(main_entrypoint, ctx);
free(ctx);
;

#pragma omp_to_hclib body_end
    long long end_time = get_time();

    printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result.txt","w");
    fprintf(fpo, "print traceback value GPU:\n");

    for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
        int nw, n, w, traceback;
        if ( i == max_rows - 2 && j == max_rows - 2 )
            fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]); //print the first element
        if ( i == 0 && j == 0 )
            break;
        if ( i > 0 && j > 0 ){
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
        }
        else if ( i == 0 ){
            nw = n = LIMIT;
            w  = input_itemsets[ i * max_cols + j - 1 ];
        }
        else if ( j == 0 ){
            nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
        }
        else{
        }

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if(traceback == new_nw)
            traceback = nw;
        if(traceback == new_w)
            traceback = w;
        if(traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);

        if(traceback == nw )
        {i--; j--; continue;}

        else if(traceback == w )
        {j--; continue;}

        else if(traceback == n )
        {i--; continue;}

        else
            ;
    }

    fclose(fpo);

#endif

    free(referrence);
    free(input_itemsets);
    free(output_itemsets);

}



