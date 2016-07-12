#include <sys/time.h>
#include <time.h>
#include <stdio.h>
static unsigned long long current_time_ns() {
#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    unsigned long long s = 1000000000ULL * (unsigned long long)mts.tv_sec;
    return (unsigned long long)mts.tv_nsec + s;
#else
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (((unsigned long long)t.tv_nsec)) + s;
#endif
}
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

#pragma omp task untied firstprivate(n, csols, i, j, a, depth, solutions)
{
	  		/* allocate a temporary array and copy <a> into it */
	  		char * b = (char *)malloc(n * sizeof(char));
	  		memcpy(b, a, j * sizeof(char));
	  		b[j] = (char) i;
	  		if (ok(j + 1, b))
       				nqueens(n, j + 1, b,&csols[i],depth); //FIXME: depth or depth+1 ???
		}
	}

#pragma omp taskwait 
;
	for ( i = 0; i < n; i++) *solutions += csols[i];
    free(csols);
}

void find_queens (int size)
{
const unsigned long long full_program_start = current_time_ns();
{
	total_count=0;

        bots_message("Computing N-Queens algorithm (n=%d) ", size);
#pragma omp parallel 
{
#pragma omp single 
{
			char *a;

			a = (char *)malloc(size * sizeof(char));
			nqueens(size, 0, a, &total_count,0);
		}
	}
	bots_message(" completed!\n");
    } ; 
const unsigned long long full_program_end = current_time_ns();
printf("full_program %llu ns", full_program_end - full_program_start);

}


int verify_queens (int size)
{
	if ( size > MAX_SOLUTIONS ) return BOTS_RESULT_NA;
	if ( total_count == solutions[size-1]) return BOTS_RESULT_SUCCESSFUL;
	return BOTS_RESULT_UNSUCCESSFUL;
}
