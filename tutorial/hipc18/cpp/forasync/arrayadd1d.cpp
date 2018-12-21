/*
 * Addition of two vectors in parallel using forasync1D
 */

#include "hclib_cpp.h"

using namespace std;

/* Verification of computation */
void check(int *a,int val,int num_iters){
	int i;
	for(i=0;i< num_iters;i++){
		if(a[i]!=val){
			printf("ERROR a[%d]=%d!=%d\n",i,a[i],val);
			exit(0);
		}
	}
}

void taskMain(int num_iters) {
  int i;
  int *a,*b,*c;
       
  a=(int*)malloc(sizeof(int)*num_iters);
  b=(int*)malloc(sizeof(int)*num_iters);
  c=(int*)malloc(sizeof(int)*num_iters);

  /*Initialize the Values */
  for(i=0; i<num_iters; i++){
    a[i]=0;
    b[i]=100;
    c[i]=1;
  }
  /*start timer */
  long start = hclib_current_time_ms();
  /*Add the elements of arrays b and c and store them in a */
  hclib::finish([=]() {
    hclib::loop_domain_1d *loop = new hclib::loop_domain_1d(num_iters);
    hclib::forasync1D(loop, [=](int i) {
      a[i]=b[i]+c[i];
    }, false, FORASYNC_MODE_RECURSIVE);
  });
  /*end timer*/
  long end = hclib_current_time_ms();
  double dur = ((double)(end-start))/1000;

  check(a,101,num_iters);
  printf("Test passed\n");
  printf("Time = %.3f\n",dur);
}

int main(int argc, char *argv[])
{
  int num_iters = argc>1? atoi(argv[1]) : 10000000;
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    taskMain(num_iters);
  });
  return 0;
}
