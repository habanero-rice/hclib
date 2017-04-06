/*
 * This benchmark has been modified from the original version
 * to use the HClib API and runtime.
 *
 * Modifications copyright 2017 Rice University
 *
 * The original copyright notice is as follows:
 * -------------------------------------------------------------
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
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

#include "hclib_cpp.h"
#include<sys/time.h>
#include<stdlib.h>

/* MERGESIZE must be >= 2 */
#define KILO 1024
#define MERGESIZE (2*KILO)
#define QUICKSIZE (2*KILO)

int *array, *tmpArr, *back;

int partition(int left, int right) {
  int i = left;
  int j = right;
  int tmpx;
  int pivot = array[(left + right) / 2];
  while(i <= j){
    while(array[i] < pivot)
      i++;
    while(array[j] > pivot)
      j--;
    if(i <= j) {
      tmpx = array[i];
      array[i] = array[j];
      array[j] = tmpx;
      i++;
      j--;
    }
  }
  return i;
}

void quicksort(int left, int right) {
  int index = partition(left, right);
  if(left < index - 1) 
    quicksort(left, index - 1);
  if(index < right) 
    quicksort(index, right);
}

void seqmerge(int low1, int high1, int low2, int high2, int lowdest, int* src, int* dest) {
  int a1;
  int a2;
  if(low1 < high1 && low2 < high2) {
    a1 = src[low1];
    a2 = src[low2];
    while(1) {
      if(a1 < a2) {
        dest[lowdest++] = a1;
        a1 = src[++low1];
        if(low1 >= high1) 
          break ;
      }
      else {
        dest[lowdest++] = a2;
        a2 = dest[++low2];
        if(low2 >= high2) 
        break ;
      }
    }
  }
  if(low1 <= high1 && low2 <= high2) {
    a1 = src[low1];
    a2 = src[low2];
    while(1) {
      if(a1 < a2) {
        dest[lowdest++] = a1;
        ++low1;
        if(low1 > high1) 
          break ;
        a1 = src[low1];
      }
      else {
        dest[lowdest++] = a2;
        ++low2;
        if(low2 > high2) 
          break ;
        a2 = src[low2];
      }
    }
  }
  if(low1 > high1) {
    memcpy(dest+lowdest, src+low2, sizeof(int) * (high2 - low2 + 1));
  }
  else {
    memcpy(dest+lowdest, src+low1, sizeof(int) * (high1 - low1 + 1));
  }
}

int binsplit(int val, int low, int high, int* src) {
  while(low != high){
    int mid = low + ((high - low + 1) >> 1);
    if(val <= src[mid]) 
      high = mid - 1;
    else 
      low = mid;
  }
  if(src[low] > val) 
    return low - 1;
  else 
    return low;
}

void cilkmerge(int low1, int high1, int low2, int high2, int lowdest, int *src, int *dest) { 
    int split1;
    int split2;
    int lowsize;
    if(high2 - low2 > high1 - low1) {
    {
        int tmp = low1;
        low1 = low2;
        low2 = tmp;
    }
    {
        int tmp = high1;
        high1 = high2;
        high2 = tmp;
    }
    }
    if(high1 < low1) {
      memcpy(dest+lowdest, src+low2,sizeof(int) * (high2 - low2));
      return ;
    }
    if(high2 - low2 < MERGESIZE) {
      seqmerge(low1, high1, low2, high2, lowdest, dest, src);
      return ;
    }
    split1 = ((high1 - low1 + 1) / 2) + low1;
    split2 = binsplit(split1, low2, high2, src);
    lowsize = split1 - low1 + split2 - low2;
    dest[(lowdest + lowsize + 1)] = src[split1];

    hclib::finish([=]() {
      hclib::async([=]() {
        cilkmerge(low1, split1 - 1, low2, split2, lowdest, src, dest);
      });
      cilkmerge(split1 + 1, high1, split2 + 1, high2, lowdest + lowsize + 2, src, dest);
    });
}

void cilksort(int low, int tmpx, int size) {
    int quarter = size / 4;
    int A;
    int B;
    int C;
    int D;
    int tmpA;
    int tmpB;
    int tmpC;
    int tmpD;
    if(size < QUICKSIZE) {
      quicksort(low, low + size - 1);
      return ;
    }
    A = low;
    tmpA = tmpx;
    B = A + quarter;
    tmpB = tmpA + quarter;
    C = B + quarter;
    tmpC = tmpB + quarter;
    D = C + quarter;
    tmpD = tmpC + quarter;

    hclib::finish([=]() {
      hclib::async([=]() {
        cilksort(A, tmpA, quarter);
      });
      hclib::async([=]() {
        cilksort(B, tmpB, quarter);
      });
      hclib::async([=]() {
        cilksort(C, tmpC, quarter);
      });
      cilksort(D, tmpD, size - 3 * quarter);
    });

    hclib::finish([=]() {
      hclib::async([=]() {
        cilkmerge(A, A + quarter - 1, B, B + quarter - 1, tmpA, array, tmpArr);
      });
      cilkmerge(C, C + quarter - 1, D, low + size - 1, tmpC, array, tmpArr);
    });

    cilkmerge(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A, tmpArr, array);
}

long get_usecs (void)
{
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

#define swap(a, b) \
{ \
  int tmp;\
  tmp = a;\
  a = b;\
  b = tmp;\
}

void scramble_array(int *arr, int size)
{
     int i;

     for (i = 0; i < size; ++i) {
	  int j = rand();
	  j = j % size;
	  swap(arr[i], arr[j]);
     }
}

int main(int argc, char **argv)
{
     hclib::launch([&]() {
         int size = 10000000;
         int i, k;
         
         if(argc > 1) size = atoi(argv[1]);
         
         array = (int*) malloc(size * sizeof(int));
         back = (int*) malloc(size * sizeof(int));
         tmpArr = (int*) malloc(size * sizeof(int));
      
         srand(1);
         for(i=0; i<size; i++) {
           back[i] = i;
         }
         scramble_array(back,size);
        
         long start = get_usecs();
         memcpy(array, back,sizeof(int) * size);   
         cilksort(0, 0, size);
         long end = get_usecs();
         double dur = ((double)(end-start))/1000000;
         int passed = 0;
         for (k = 0; k < size; ++k)
        if (array[k] != k)
                passed = 1;

         printf("CilkSort (%d): Passed = %d, Time = %f\n",size,passed,dur);

        free(array);
        free(back);
        free(tmpArr);
    });

    return 0;
}
