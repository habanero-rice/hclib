#include "hclib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "timer.h"

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

#define BENCH_PRINT

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9

void
init(int argc, char** argv)
{
	if(argc==3){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	}else{
                printf("Usage: pathfiner width num_of_steps\n");
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

typedef struct _run99 {
    int argc;
    char **argv;
    unsigned long long cycles;
    int *src;
    int *dst;
    int *temp;
    int min;
    int t;
 } run99;

static void run99_hclib_async(void *arg, const int ___iter) {
    run99 *ctx = (run99 *)arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    unsigned long long cycles; cycles = ctx->cycles;
    int *src; src = ctx->src;
    int *dst; dst = ctx->dst;
    int *temp; temp = ctx->temp;
    int min; min = ctx->min;
    int t; t = ctx->t;
    hclib_start_finish();
    do {
    int n;     n = ___iter;
{
          min = src[n];
          if (n > 0)
            min = MIN(min, src[n-1]);
          if (n < cols-1)
            min = MIN(min, src[n+1]);
          dst[n] = wall[t+1][n]+min;
        }    } while (0);
    hclib_end_finish();
}

typedef struct _main_entrypoint_ctx {
    int argc;
    char **argv;
    unsigned long long cycles;
    int *src;
    int *dst;
    int *temp;
    int min;
 } main_entrypoint_ctx;

static void main_entrypoint(void *arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    unsigned long long cycles; cycles = ctx->cycles;
    int *src; src = ctx->src;
    int *dst; dst = ctx->dst;
    int *temp; temp = ctx->temp;
    int min; min = ctx->min;
for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
         { 
run99 *ctx = (run99 *)malloc(sizeof(run99));
ctx->argc = argc;
ctx->argv = argv;
ctx->cycles = cycles;
ctx->src = src;
ctx->dst = dst;
ctx->temp = temp;
ctx->min = min;
ctx->t = t;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = cols;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)run99_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
    }; }

void run(int argc, char** argv)
{
    init(argc, argv);

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();
#pragma omp_to_hclib body_start
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->argc = argc;
ctx->argv = argv;
ctx->cycles = cycles;
ctx->src = src;
ctx->dst = dst;
ctx->temp = temp;
ctx->min = min;
hclib_launch(main_entrypoint, ctx);
free(ctx);

#pragma omp_to_hclib body_end

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",dst[i]) ;
    printf("\n") ;
#endif

    delete [] data;
    delete [] wall;
    delete [] dst;
    delete [] src;
}

