#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

/*
 * Original code from the Cilk project (by Keith Randall)
 * 
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "bots.h"
#include "app-desc.h"
#include <omp.h>


/* Checking information */

static int solutions[] = {
        1,
        0,
        0,
        2,
        10, /* 5 */
        4,
        40,
        92,
        352,
        724, /* 10 */
        2680,
        14200,
        73712,
        365596,
};
#define MAX_SOLUTIONS sizeof(solutions)/sizeof(int)

int total_count;


/*
 * <a> contains array of <n> queen positions.  Returns 1
 * if none of the queens conflict, and returns 0 otherwise.
 */
int ok(int n, char *a)
{
     int i, j;
     char p, q;

     for (i = 0; i < n; i++) {
	  p = a[i];

	  for (j = i + 1; j < n; j++) {
	       q = a[j];
	       if (q == p || q == p - (j - i) || q == p + (j - i))
		    return 0;
	  }
     }
     return 1;
}

void nqueens_ser (int n, int j, char *a, int *solutions)
{
	int res;
	int i;

	if (n == j) {
		/* good solution, count it */
		*solutions = 1;
		return;
	}

	*solutions = 0;

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
		{
	  		/* allocate a temporary array and copy <a> into it */
	  		a[j] = (char) i;
	  		if (ok(j + 1, a)) {
	       			nqueens_ser(n, j + 1, a,&res);
				*solutions += res;
			}
		}
	}
}

typedef struct _pragma126_omp_task {
    int (*csols);
    int i;
    int n;
    int j;
    char (*a);
    int (*solutions);
    int depth;
 } pragma126_omp_task;

static void pragma126_omp_task_hclib_async(void *____arg);
void nqueens(int n, int j, char *a, int *solutions, int depth)
{
	int *csols;
	int i;


	if (n == j) {
		/* good solution, count it */
		*solutions = 1;
		return;
	}


	*solutions = 0;
	csols = (int *)malloc(n*sizeof(int));
	memset(csols,0,n*sizeof(int));

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {

 { 
pragma126_omp_task *new_ctx = (pragma126_omp_task *)malloc(sizeof(pragma126_omp_task));
new_ctx->csols = csols;
new_ctx->i = i;
new_ctx->n = n;
new_ctx->j = j;
new_ctx->a = a;
new_ctx->solutions = solutions;
new_ctx->depth = depth;
hclib_async(pragma126_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
	}

 hclib_end_finish(); hclib_start_finish(); ;
	for ( i = 0; i < n; i++) *solutions += csols[i];
    free(csols);
} 
static void pragma126_omp_task_hclib_async(void *____arg) {
    pragma126_omp_task *ctx = (pragma126_omp_task *)____arg;
    int (*csols); csols = ctx->csols;
    int i; i = ctx->i;
    int n; n = ctx->n;
    int j; j = ctx->j;
    char (*a); a = ctx->a;
    int (*solutions); solutions = ctx->solutions;
    int depth; depth = ctx->depth;
    hclib_start_finish();
{
	  		/* allocate a temporary array and copy <a> into it */
	  		char * b = (char *)malloc(n * sizeof(char));
	  		memcpy(b, a, j * sizeof(char));
	  		b[j] = (char) i;
	  		if (ok(j + 1, b))
       				nqueens(n, j + 1, b,&csols[i],depth); //FIXME: depth or depth+1 ???
		} ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}



typedef struct _main_entrypoint_ctx {
    int size;
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int size; size = ctx->size;
{
	total_count=0;

        bots_message("Computing N-Queens algorithm (n=%d) ", size);
hclib_start_finish(); {
			char *a;

			a = (char *)malloc(size * sizeof(char));
			nqueens(size, 0, a, &total_count,0);
		} ; hclib_end_finish(); 
	bots_message(" completed!\n");
    } ;     free(____arg);
}

void find_queens (int size)
{
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->size = size;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);

} 


int verify_queens (int size)
{
	if ( size > MAX_SOLUTIONS ) return BOTS_RESULT_NA;
	if ( total_count == solutions[size-1]) return BOTS_RESULT_SUCCESSFUL;
	return BOTS_RESULT_UNSUCCESSFUL;
}
