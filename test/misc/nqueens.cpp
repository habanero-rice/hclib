#include <iostream>
#include <stdio.h>
#include "hcpp.h"
#include <sys/time.h>
using namespace hcpp;

int solutions[16] =
{
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
2279184, 
14772512
};

volatile int *atomic;

int ok(int n,  int* A) {
  int i, j;
  for (i =  0; i < n; i++) {
    int p = A[i];

    for (j =  (i +  1); j < n; j++) {
      int q = A[j];
      if (q == p || q == p - (j - i) || q == p + (j - i))
      return 1;
    }
  }
  return 0;
}

void nqueens_kernel(int* A, int depth, int size) {
  if (size == depth) {
    __sync_fetch_and_add(atomic, 1);
    return;
  }
  /* try each possible position for queen <depth> */
  int i =0;
  hcpp::finish ([&](){
    hcpp::forasync(0, size, [=](int i) {
      /* allocate a temporary array and copy <a> into it */
      int* B = (int*) malloc(sizeof(int)*(depth+1));
      memcpy(B, A, sizeof(int)*depth);
      B[depth] = i;
      int failed = ok((depth +  1), B); 
      if (!failed) {
        nqueens_kernel(B, depth+1, size);
      }
    });
  });
  free(A);
}

void verify_queens(int size) {
  if ( *atomic == solutions[size-1] )
    printf("OK\n");
   else
    printf("Incorrect Answer\n");
}

long get_usecs (void)
{
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

int main(int argc, char* argv[])
{
  int n = 12;
  int i, j;
     
  if(argc > 1) n = atoi(argv[1]);
     
  double dur = 0;
  int* a = (int*) malloc(sizeof(int));
  atomic = (int*) malloc(sizeof(int));;
  atomic[0]=0;
  long start = get_usecs();
  nqueens_kernel(a, 0, n);  
  long end = get_usecs();
  dur = ((double)(end-start))/1000000;
  verify_queens(n);  
  free((void*)atomic);
  printf("NQueens(%d) Time = %fsec\n",n,dur);
  return 0;
}
