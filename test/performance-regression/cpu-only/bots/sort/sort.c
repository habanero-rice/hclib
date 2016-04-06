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
 *  Original code from the Cilk project
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

/*
 * this program uses an algorithm that we call `cilksort'.
 * The algorithm is essentially mergesort:
 *
 *   cilksort(in[1..n]) =
 *       spawn cilksort(in[1..n/2], tmp[1..n/2])
 *       spawn cilksort(in[n/2..n], tmp[n/2..n])
 *       sync
 *       spawn cilkmerge(tmp[1..n/2], tmp[n/2..n], in[1..n])
 *
 *
 * The procedure cilkmerge does the following:
 *       
 *       cilkmerge(A[1..n], B[1..m], C[1..(n+m)]) =
 *          find the median of A \union B using binary
 *          search.  The binary search gives a pair
 *          (ma, mb) such that ma + mb = (n + m)/2
 *          and all elements in A[1..ma] are smaller than
 *          B[mb..m], and all the B[1..mb] are smaller
 *          than all elements in A[ma..n].
 *
 *          spawn cilkmerge(A[1..ma], B[1..mb], C[1..(n+m)/2])
 *          spawn cilkmerge(A[ma..m], B[mb..n], C[(n+m)/2 .. (n+m)])
 *          sync
 *
 * The algorithm appears for the first time (AFAIK) in S. G. Akl and
 * N. Santoro, "Optimal Parallel Merging and Sorting Without Memory
 * Conflicts", IEEE Trans. Comp., Vol. C-36 No. 11, Nov. 1987 .  The
 * paper does not express the algorithm using recursion, but the
 * idea of finding the median is there.
 *
 * For cilksort of n elements, T_1 = O(n log n) and
 * T_\infty = O(log^3 n).  There is a way to shave a
 * log factor in the critical path (left as homework).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bots.h"
#include "app-desc.h"

ELM *array, *tmp;

static unsigned long rand_nxt = 0;

static inline unsigned long my_rand(void)
{
     rand_nxt = rand_nxt * 1103515245 + 12345;
     return rand_nxt;
}

static inline void my_srand(unsigned long seed)
{
     rand_nxt = seed;
}

static inline ELM med3(ELM a, ELM b, ELM c)
{
     if (a < b) {
	  if (b < c) {
	       return b;
	  } else {
	       if (a < c)
		    return c;
	       else
		    return a;
	  }
     } else {
	  if (b > c) {
	       return b;
	  } else {
	       if (a > c)
		    return c;
	       else
		    return a;
	  }
     }
}

/*
 * simple approach for now; a better median-finding
 * may be preferable
 */
static inline ELM choose_pivot(ELM *low, ELM *high)
{
     return med3(*low, *high, low[(high - low) / 2]);
}

static ELM *seqpart(ELM *low, ELM *high)
{
     ELM pivot;
     ELM h, l;
     ELM *curr_low = low;
     ELM *curr_high = high;

     pivot = choose_pivot(low, high);

     while (1) {
	  while ((h = *curr_high) > pivot)
	       curr_high--;

	  while ((l = *curr_low) < pivot)
	       curr_low++;

	  if (curr_low >= curr_high)
	       break;

	  *curr_high-- = l;
	  *curr_low++ = h;
     }

     /*
      * I don't know if this is really necessary.
      * The problem is that the pivot is not always the
      * first element, and the partition may be trivial.
      * However, if the partition is trivial, then
      * *high is the largest element, whence the following
      * code.
      */
     if (curr_high < high)
	  return curr_high;
     else
	  return curr_high - 1;
}

#define swap(a, b) \
{ \
  ELM tmp;\
  tmp = a;\
  a = b;\
  b = tmp;\
}

static void insertion_sort(ELM *low, ELM *high)
{
     ELM *p, *q;
     ELM a, b;

     for (q = low + 1; q <= high; ++q) {
	  a = q[0];
	  for (p = q - 1; p >= low && (b = p[0]) > a; p--)
	       p[1] = b;
	  p[1] = a;
     }
}

/*
 * tail-recursive quicksort, almost unrecognizable :-)
 */
void seqquick(ELM *low, ELM *high)
{
     ELM *p;

     while (high - low >= bots_app_cutoff_value_2) {
	  p = seqpart(low, high);
	  seqquick(low, p);
	  low = p + 1;
     }

     insertion_sort(low, high);
}

void seqmerge(ELM *low1, ELM *high1, ELM *low2, ELM *high2,
	      ELM *lowdest)
{
     ELM a1, a2;

     /*
      * The following 'if' statement is not necessary
      * for the correctness of the algorithm, and is
      * in fact subsumed by the rest of the function.
      * However, it is a few percent faster.  Here is why.
      *
      * The merging loop below has something like
      *   if (a1 < a2) {
      *        *dest++ = a1;
      *        ++low1;
      *        if (end of array) break;
      *        a1 = *low1;
      *   }
      *
      * Now, a1 is needed immediately in the next iteration
      * and there is no way to mask the latency of the load.
      * A better approach is to load a1 *before* the end-of-array
      * check; the problem is that we may be speculatively
      * loading an element out of range.  While this is
      * probably not a problem in practice, yet I don't feel
      * comfortable with an incorrect algorithm.  Therefore,
      * I use the 'fast' loop on the array (except for the last 
      * element) and the 'slow' loop for the rest, saving both
      * performance and correctness.
      */

     if (low1 < high1 && low2 < high2) {
	  a1 = *low1;
	  a2 = *low2;
	  for (;;) {
	       if (a1 < a2) {
		    *lowdest++ = a1;
		    a1 = *++low1;
		    if (low1 >= high1)
			 break;
	       } else {
		    *lowdest++ = a2;
		    a2 = *++low2;
		    if (low2 >= high2)
			 break;
	       }
	  }
     }
     if (low1 <= high1 && low2 <= high2) {
	  a1 = *low1;
	  a2 = *low2;
	  for (;;) {
	       if (a1 < a2) {
		    *lowdest++ = a1;
		    ++low1;
		    if (low1 > high1)
			 break;
		    a1 = *low1;
	       } else {
		    *lowdest++ = a2;
		    ++low2;
		    if (low2 > high2)
			 break;
		    a2 = *low2;
	       }
	  }
     }
     if (low1 > high1) {
	  memcpy(lowdest, low2, sizeof(ELM) * (high2 - low2 + 1));
     } else {
	  memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1 + 1));
     }
}

#define swap_indices(a, b) \
{ \
  ELM *tmp;\
  tmp = a;\
  a = b;\
  b = tmp;\
}

ELM *binsplit(ELM val, ELM *low, ELM *high)
{
     /*
      * returns index which contains greatest element <= val.  If val is
      * less than all elements, returns low-1
      */
     ELM *mid;

     while (low != high) {
	  mid = low + ((high - low + 1) >> 1);
	  if (val <= *mid)
	       high = mid - 1;
	  else
	       low = mid;
     }

     if (*low > val)
	  return low - 1;
     else
	  return low;
}


typedef struct _pragma350 {
    ELM (*(*split1_ptr));
    ELM (*(*split2_ptr));
    long (*lowsize_ptr);
    ELM (*(*low1_ptr));
    ELM (*(*high1_ptr));
    ELM (*(*low2_ptr));
    ELM (*(*high2_ptr));
    ELM (*(*lowdest_ptr));
 } pragma350;

typedef struct _pragma352 {
    ELM (*(*split1_ptr));
    ELM (*(*split2_ptr));
    long (*lowsize_ptr);
    ELM (*(*low1_ptr));
    ELM (*(*high1_ptr));
    ELM (*(*low2_ptr));
    ELM (*(*high2_ptr));
    ELM (*(*lowdest_ptr));
 } pragma352;

static void pragma350_hclib_async(void *____arg);
static void pragma352_hclib_async(void *____arg);
void cilkmerge_par(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest)
{
     /*
      * Cilkmerge: Merges range [low1, high1] with range [low2, high2] 
      * into the range [lowdest, ...]  
      */

     ELM *split1, *split2;	/*
				 * where each of the ranges are broken for 
				 * recursive merge 
				 */
     long int lowsize;		/*
				 * total size of lower halves of two
				 * ranges - 2 
				 */

     /*
      * We want to take the middle element (indexed by split1) from the
      * larger of the two arrays.  The following code assumes that split1
      * is taken from range [low1, high1].  So if [low1, high1] is
      * actually the smaller range, we should swap it with [low2, high2] 
      */

     if (high2 - low2 > high1 - low1) {
	  swap_indices(low1, low2);
	  swap_indices(high1, high2);
     }
     if (high2 < low2) {
	  /* smaller range is empty */
	  memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1));
	  return;
     }
     if (high2 - low2 < bots_app_cutoff_value ) {
	  seqmerge(low1, high1, low2, high2, lowdest);
	  return;
     }
     /*
      * Basic approach: Find the middle element of one range (indexed by
      * split1). Find where this element would fit in the other range
      * (indexed by split 2). Then merge the two lower halves and the two
      * upper halves. 
      */

     split1 = ((high1 - low1 + 1) / 2) + low1;
     split2 = binsplit(*split1, low2, high2);
     lowsize = split1 - low1 + split2 - low2;

     /* 
      * directly put the splitting element into
      * the appropriate location
      */
     *(lowdest + lowsize + 1) = *split1;
 { 
pragma350 *new_ctx = (pragma350 *)malloc(sizeof(pragma350));
new_ctx->split1_ptr = &(split1);
new_ctx->split2_ptr = &(split2);
new_ctx->lowsize_ptr = &(lowsize);
new_ctx->low1_ptr = &(low1);
new_ctx->high1_ptr = &(high1);
new_ctx->low2_ptr = &(low2);
new_ctx->high2_ptr = &(high2);
new_ctx->lowdest_ptr = &(lowdest);
hclib_async(pragma350_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma352 *new_ctx = (pragma352 *)malloc(sizeof(pragma352));
new_ctx->split1_ptr = &(split1);
new_ctx->split2_ptr = &(split2);
new_ctx->lowsize_ptr = &(lowsize);
new_ctx->low1_ptr = &(low1);
new_ctx->high1_ptr = &(high1);
new_ctx->low2_ptr = &(low2);
new_ctx->high2_ptr = &(high2);
new_ctx->lowdest_ptr = &(lowdest);
hclib_async(pragma352_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;

     return;
} 
static void pragma350_hclib_async(void *____arg) {
    pragma350 *ctx = (pragma350 *)____arg;
    hclib_start_finish();
cilkmerge_par((*(ctx->low1_ptr)), (*(ctx->split1_ptr)) - 1, (*(ctx->low2_ptr)), (*(ctx->split2_ptr)), (*(ctx->lowdest_ptr))) ;     ; hclib_end_finish();

}


static void pragma352_hclib_async(void *____arg) {
    pragma352 *ctx = (pragma352 *)____arg;
    hclib_start_finish();
cilkmerge_par((*(ctx->split1_ptr)) + 1, (*(ctx->high1_ptr)), (*(ctx->split2_ptr)) + 1, (*(ctx->high2_ptr)),
		     (*(ctx->lowdest_ptr)) + (*(ctx->lowsize_ptr)) + 2) ;     ; hclib_end_finish();

}



typedef struct _pragma386 {
    long (*quarter_ptr);
    ELM (*(*A_ptr));
    ELM (*(*B_ptr));
    ELM (*(*C_ptr));
    ELM (*(*D_ptr));
    ELM (*(*tmpA_ptr));
    ELM (*(*tmpB_ptr));
    ELM (*(*tmpC_ptr));
    ELM (*(*tmpD_ptr));
    ELM (*(*low_ptr));
    ELM (*(*tmp_ptr));
    long (*size_ptr);
 } pragma386;

typedef struct _pragma388 {
    long (*quarter_ptr);
    ELM (*(*A_ptr));
    ELM (*(*B_ptr));
    ELM (*(*C_ptr));
    ELM (*(*D_ptr));
    ELM (*(*tmpA_ptr));
    ELM (*(*tmpB_ptr));
    ELM (*(*tmpC_ptr));
    ELM (*(*tmpD_ptr));
    ELM (*(*low_ptr));
    ELM (*(*tmp_ptr));
    long (*size_ptr);
 } pragma388;

typedef struct _pragma390 {
    long (*quarter_ptr);
    ELM (*(*A_ptr));
    ELM (*(*B_ptr));
    ELM (*(*C_ptr));
    ELM (*(*D_ptr));
    ELM (*(*tmpA_ptr));
    ELM (*(*tmpB_ptr));
    ELM (*(*tmpC_ptr));
    ELM (*(*tmpD_ptr));
    ELM (*(*low_ptr));
    ELM (*(*tmp_ptr));
    long (*size_ptr);
 } pragma390;

typedef struct _pragma392 {
    long (*quarter_ptr);
    ELM (*(*A_ptr));
    ELM (*(*B_ptr));
    ELM (*(*C_ptr));
    ELM (*(*D_ptr));
    ELM (*(*tmpA_ptr));
    ELM (*(*tmpB_ptr));
    ELM (*(*tmpC_ptr));
    ELM (*(*tmpD_ptr));
    ELM (*(*low_ptr));
    ELM (*(*tmp_ptr));
    long (*size_ptr);
 } pragma392;

typedef struct _pragma396 {
    long (*quarter_ptr);
    ELM (*(*A_ptr));
    ELM (*(*B_ptr));
    ELM (*(*C_ptr));
    ELM (*(*D_ptr));
    ELM (*(*tmpA_ptr));
    ELM (*(*tmpB_ptr));
    ELM (*(*tmpC_ptr));
    ELM (*(*tmpD_ptr));
    ELM (*(*low_ptr));
    ELM (*(*tmp_ptr));
    long (*size_ptr);
 } pragma396;

typedef struct _pragma398 {
    long (*quarter_ptr);
    ELM (*(*A_ptr));
    ELM (*(*B_ptr));
    ELM (*(*C_ptr));
    ELM (*(*D_ptr));
    ELM (*(*tmpA_ptr));
    ELM (*(*tmpB_ptr));
    ELM (*(*tmpC_ptr));
    ELM (*(*tmpD_ptr));
    ELM (*(*low_ptr));
    ELM (*(*tmp_ptr));
    long (*size_ptr);
 } pragma398;

static void pragma386_hclib_async(void *____arg);
static void pragma388_hclib_async(void *____arg);
static void pragma390_hclib_async(void *____arg);
static void pragma392_hclib_async(void *____arg);
static void pragma396_hclib_async(void *____arg);
static void pragma398_hclib_async(void *____arg);
void cilksort_par(ELM *low, ELM *tmp, long size)
{
     /*
      * divide the input in four parts of the same size (A, B, C, D)
      * Then:
      *   1) recursively sort A, B, C, and D (in parallel)
      *   2) merge A and B into tmp1, and C and D into tmp2 (in parallel)
      *   3) merge tmp1 and tmp2 into the original array
      */
     long quarter = size / 4;
     ELM *A, *B, *C, *D, *tmpA, *tmpB, *tmpC, *tmpD;

     if (size < bots_app_cutoff_value_1 ) {
	  /* quicksort when less than 1024 elements */
	  seqquick(low, low + size - 1);
	  return;
     }
     A = low;
     tmpA = tmp;
     B = A + quarter;
     tmpB = tmpA + quarter;
     C = B + quarter;
     tmpC = tmpB + quarter;
     D = C + quarter;
     tmpD = tmpC + quarter;

 { 
pragma386 *new_ctx = (pragma386 *)malloc(sizeof(pragma386));
new_ctx->quarter_ptr = &(quarter);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->D_ptr = &(D);
new_ctx->tmpA_ptr = &(tmpA);
new_ctx->tmpB_ptr = &(tmpB);
new_ctx->tmpC_ptr = &(tmpC);
new_ctx->tmpD_ptr = &(tmpD);
new_ctx->low_ptr = &(low);
new_ctx->tmp_ptr = &(tmp);
new_ctx->size_ptr = &(size);
hclib_async(pragma386_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma388 *new_ctx = (pragma388 *)malloc(sizeof(pragma388));
new_ctx->quarter_ptr = &(quarter);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->D_ptr = &(D);
new_ctx->tmpA_ptr = &(tmpA);
new_ctx->tmpB_ptr = &(tmpB);
new_ctx->tmpC_ptr = &(tmpC);
new_ctx->tmpD_ptr = &(tmpD);
new_ctx->low_ptr = &(low);
new_ctx->tmp_ptr = &(tmp);
new_ctx->size_ptr = &(size);
hclib_async(pragma388_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma390 *new_ctx = (pragma390 *)malloc(sizeof(pragma390));
new_ctx->quarter_ptr = &(quarter);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->D_ptr = &(D);
new_ctx->tmpA_ptr = &(tmpA);
new_ctx->tmpB_ptr = &(tmpB);
new_ctx->tmpC_ptr = &(tmpC);
new_ctx->tmpD_ptr = &(tmpD);
new_ctx->low_ptr = &(low);
new_ctx->tmp_ptr = &(tmp);
new_ctx->size_ptr = &(size);
hclib_async(pragma390_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma392 *new_ctx = (pragma392 *)malloc(sizeof(pragma392));
new_ctx->quarter_ptr = &(quarter);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->D_ptr = &(D);
new_ctx->tmpA_ptr = &(tmpA);
new_ctx->tmpB_ptr = &(tmpB);
new_ctx->tmpC_ptr = &(tmpC);
new_ctx->tmpD_ptr = &(tmpD);
new_ctx->low_ptr = &(low);
new_ctx->tmp_ptr = &(tmp);
new_ctx->size_ptr = &(size);
hclib_async(pragma392_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;

 { 
pragma396 *new_ctx = (pragma396 *)malloc(sizeof(pragma396));
new_ctx->quarter_ptr = &(quarter);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->D_ptr = &(D);
new_ctx->tmpA_ptr = &(tmpA);
new_ctx->tmpB_ptr = &(tmpB);
new_ctx->tmpC_ptr = &(tmpC);
new_ctx->tmpD_ptr = &(tmpD);
new_ctx->low_ptr = &(low);
new_ctx->tmp_ptr = &(tmp);
new_ctx->size_ptr = &(size);
hclib_async(pragma396_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma398 *new_ctx = (pragma398 *)malloc(sizeof(pragma398));
new_ctx->quarter_ptr = &(quarter);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->D_ptr = &(D);
new_ctx->tmpA_ptr = &(tmpA);
new_ctx->tmpB_ptr = &(tmpB);
new_ctx->tmpC_ptr = &(tmpC);
new_ctx->tmpD_ptr = &(tmpD);
new_ctx->low_ptr = &(low);
new_ctx->tmp_ptr = &(tmp);
new_ctx->size_ptr = &(size);
hclib_async(pragma398_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;

     cilkmerge_par(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A);
} 
static void pragma386_hclib_async(void *____arg) {
    pragma386 *ctx = (pragma386 *)____arg;
    hclib_start_finish();
cilksort_par((*(ctx->A_ptr)), (*(ctx->tmpA_ptr)), (*(ctx->quarter_ptr))) ;     ; hclib_end_finish();

}


static void pragma388_hclib_async(void *____arg) {
    pragma388 *ctx = (pragma388 *)____arg;
    hclib_start_finish();
cilksort_par((*(ctx->B_ptr)), (*(ctx->tmpB_ptr)), (*(ctx->quarter_ptr))) ;     ; hclib_end_finish();

}


static void pragma390_hclib_async(void *____arg) {
    pragma390 *ctx = (pragma390 *)____arg;
    hclib_start_finish();
cilksort_par((*(ctx->C_ptr)), (*(ctx->tmpC_ptr)), (*(ctx->quarter_ptr))) ;     ; hclib_end_finish();

}


static void pragma392_hclib_async(void *____arg) {
    pragma392 *ctx = (pragma392 *)____arg;
    hclib_start_finish();
cilksort_par((*(ctx->D_ptr)), (*(ctx->tmpD_ptr)), (*(ctx->size_ptr)) - 3 * (*(ctx->quarter_ptr))) ;     ; hclib_end_finish();

}


static void pragma396_hclib_async(void *____arg) {
    pragma396 *ctx = (pragma396 *)____arg;
    hclib_start_finish();
cilkmerge_par((*(ctx->A_ptr)), (*(ctx->A_ptr)) + (*(ctx->quarter_ptr)) - 1, (*(ctx->B_ptr)), (*(ctx->B_ptr)) + (*(ctx->quarter_ptr)) - 1, (*(ctx->tmpA_ptr))) ;     ; hclib_end_finish();

}


static void pragma398_hclib_async(void *____arg) {
    pragma398 *ctx = (pragma398 *)____arg;
    hclib_start_finish();
cilkmerge_par((*(ctx->C_ptr)), (*(ctx->C_ptr)) + (*(ctx->quarter_ptr)) - 1, (*(ctx->D_ptr)), (*(ctx->low_ptr)) + (*(ctx->size_ptr)) - 1, (*(ctx->tmpC_ptr))) ;     ; hclib_end_finish();

}



void scramble_array( ELM *array )
{
     unsigned long i;
     unsigned long j;

     for (i = 0; i < bots_arg_size; ++i) {
	  j = my_rand();
	  j = j % bots_arg_size;
	  swap(array[i], array[j]);
     }
}

void fill_array( ELM *array )
{
     unsigned long i;

     my_srand(1);
     /* first, fill with integers 1..size */
     for (i = 0; i < bots_arg_size; ++i) {
	  array[i] = i;
     }
}

void sort_init ( void )
{
     /* Checking arguments */
     if (bots_arg_size < 4) {
        bots_message("%s can not be less than 4, using 4 as a parameter.\n", BOTS_APP_DESC_ARG_SIZE );
        bots_arg_size = 4;
     }

     if (bots_app_cutoff_value < 2) {
        bots_message("%s can not be less than 2, using 2 as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF);
        bots_app_cutoff_value = 2;
     }
     else if (bots_app_cutoff_value > bots_arg_size ) {
        bots_message("%s can not be greather than vector size, using %d as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF, bots_arg_size);
        bots_app_cutoff_value = bots_arg_size;
     }

     if (bots_app_cutoff_value_1 > bots_arg_size ) {
        bots_message("%s can not be greather than vector size, using %d as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF_1, bots_arg_size);
        bots_app_cutoff_value_1 = bots_arg_size;
     }
     if (bots_app_cutoff_value_2 > bots_arg_size ) {
        bots_message("%s can not be greather than vector size, using %d as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF_2, bots_arg_size);
        bots_app_cutoff_value_2 = bots_arg_size;
     }

     if (bots_app_cutoff_value_2 > bots_app_cutoff_value_1) {
        bots_message("%s can not be greather than %s, using %d as a parameter.\n",
		BOTS_APP_DESC_ARG_CUTOFF_2,
		BOTS_APP_DESC_ARG_CUTOFF_1,
		bots_app_cutoff_value_1
	);
        bots_app_cutoff_value_2 = bots_app_cutoff_value_1;
     }

     array = (ELM *) malloc(bots_arg_size * sizeof(ELM));
     tmp = (ELM *) malloc(bots_arg_size * sizeof(ELM));
     fill_array(array);
     scramble_array(array);
}

typedef struct _pragma478 {
 } pragma478;

static void pragma478_hclib_async(void *____arg);
typedef struct _main_entrypoint_ctx {
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
{
hclib_start_finish(); {
 { 
pragma478 *new_ctx = (pragma478 *)malloc(sizeof(pragma478));
hclib_async(pragma478_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
            } ; hclib_end_finish(); 
    } ; }

void sort_par ( void )
{
	bots_message("Computing multisort algorithm (n=%d) ", bots_arg_size);
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
hclib_launch(main_entrypoint, new_ctx);
free(new_ctx);

	bots_message(" completed!\n");
}  
static void pragma478_hclib_async(void *____arg) {
    pragma478 *ctx = (pragma478 *)____arg;
    hclib_start_finish();
{
                    cilksort_par(array, tmp, bots_arg_size);
                } ;     ; hclib_end_finish();

}



int sort_verify ( void )
{
     int i, success = 1;
     for (i = 0; i < bots_arg_size; ++i)
	  if (array[i] != i)
	       success = 0;

     return success ? BOTS_RESULT_SUCCESSFUL : BOTS_RESULT_UNSUCCESSFUL;
}

