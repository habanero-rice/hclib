/*
 * HC CONCORD foreach add.hc example 
 */

#include "hclib_cpp.h"
#include <sys/time.h>

using namespace std;

/*
 *  * Timer routine
 *   */
long get_usecs (void) {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}

void check(int *a,int val,int num_iters){
	int i;
	for(i=0;i< num_iters;i++){
		if(a[i]!=val){
			printf("ERROR a[%d]=%d!=%d\n",i,a[i],val);
			exit(0);
		}
	}
}

int main(int argc, char *argv[])
{
    char const *deps[] = { "system" }; 
    hclib::launch(deps, 1, [&]() {
        int num_iters;
        int i;
        int *a,*b,*c;
       
       if(argc!=2){printf("USAGE:./arrayadd1d SIZE\n");return -1;}
        num_iters= atoi(argv[1]);

        a=(int*)malloc(sizeof(int)*num_iters);
        b=(int*)malloc(sizeof(int)*num_iters);
        c=(int*)malloc(sizeof(int)*num_iters);

      //Initialize the Values
        for(i=0; i<num_iters; i++){

        a[i]=0;
        b[i]=100;
        c[i]=1;

        }
       //start timer
       long start = get_usecs();
       //Add the elements of arrays b and c and store them in a
       hclib::finish([=]() {
         hclib::loop_domain_1d *loop = new hclib::loop_domain_1d(num_iters);
         hclib::forasync1D(loop, [=](int i) {
            a[i]=b[i]+c[i];
         }, false, FORASYNC_MODE_RECURSIVE);
         //}, FORASYNC_MODE_FLAT);
       });
       //end timer
       long end = get_usecs();
       double dur = ((double)(end-start))/1000000;

       check(a,101,num_iters);
       printf("Test passed\n");
      printf("Time = %.3f\n",dur);
   });
   return 0;
}
