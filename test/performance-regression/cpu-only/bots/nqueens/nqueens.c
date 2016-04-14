#include "hclib.h"
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
#include <alloca.h>
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

typedef struct _pragma128 {
    int (*(*csols_ptr));
    int (*i_ptr);
    int n;
    int (*j_ptr);
    char (*(*a_ptr));
    int (*(*solutions_ptr));
    int (*depth_ptr);
 } pragma128;

static void pragma128_hclib_async(void *____arg);
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
	csols = (int *)alloca(n*sizeof(int));
	memset(csols,0,n*sizeof(int));

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
 { 
pragma128 *new_ctx = (pragma128 *)malloc(sizeof(pragma128));
new_ctx->csols_ptr = &(csols);
new_ctx->i_ptr = &(i);
new_ctx->n = n;
new_ctx->j_ptr = &(j);
new_ctx->a_ptr = &(a);
new_ctx->solutions_ptr = &(solutions);
new_ctx->depth_ptr = &(depth);
hclib_async(pragma128_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
	}

 hclib_end_finish(); hclib_start_finish(); ;
	for ( i = 0; i < n; i++) *solutions += csols[i];
} 
static void pragma128_hclib_async(void *____arg) {
    pragma128 *ctx = (pragma128 *)____arg;
    int n; n = ctx->n;
    hclib_start_finish();
{
	  		/* allocate a temporary array and copy <a> into it */
	  		char * b = (char *)alloca(n * sizeof(char));
	  		memcpy(b, (*(ctx->a_ptr)), (*(ctx->j_ptr)) * sizeof(char));
	  		b[(*(ctx->j_ptr))] = (char) (*(ctx->i_ptr));
	  		if (ok((*(ctx->j_ptr)) + 1, b))
       				nqueens(n, (*(ctx->j_ptr)) + 1, b,&(*(ctx->csols_ptr))[(*(ctx->i_ptr))],(*(ctx->depth_ptr))); //FIXME: depth or depth+1 ???
		} ;     ; hclib_end_finish();

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

			a = (char *)alloca(size * sizeof(char));
			nqueens(size, 0, a, &total_count,0);
		} ; hclib_end_finish(); 
	bots_message(" completed!\n");
    } ;     free(____arg);
}

void find_queens (int size)
{
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->size = size;
hclib_launch(main_entrypoint, new_ctx);

} 


int verify_queens (int size)
{
	if ( size > MAX_SOLUTIONS ) return BOTS_RESULT_NA;
	if ( total_count == solutions[size-1]) return BOTS_RESULT_SUCCESSFUL;
	return BOTS_RESULT_UNSUCCESSFUL;
}
