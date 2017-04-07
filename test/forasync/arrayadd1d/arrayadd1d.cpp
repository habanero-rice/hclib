/*
 * HC CONCORD foreach add.hc example 
 */

#include "hclib_cpp.h"
#include <sys/time.h>

using namespace std;

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
    hclib::launch([=]() {
        int num_iters;
        int tilesize;
        int i;
        int *a,*b,*c;
       
        if (argc != 3) {
            printf("USAGE:./arrayadd1d NUM_ITERS TILE_SIZE\n");
            exit(1);
        }
        num_iters= atoi(argv[1]);
        tilesize = atoi(argv[2]);

        a=(int*)malloc(sizeof(int)*num_iters);
        b=(int*)malloc(sizeof(int)*num_iters);
        c=(int*)malloc(sizeof(int)*num_iters);

      //Initialize the Values
        for(i=0; i<num_iters; i++){

        a[i]=0;
        b[i]=100;
        c[i]=1;

        }
       //Add the elements of arrays b and c and store them in a
       hclib::finish([=]() {
         hclib::loop_domain_t loop = {0, num_iters, 1, tilesize};
         hclib::forasync1D(&loop, [=](int i) {
            a[i]=b[i]+c[i];
         }, FORASYNC_MODE_RECURSIVE);
         //}, FORASYNC_MODE_FLAT);
       });

       check(a,101,num_iters);
       printf("Test passed\n");
   });
   return 0;
}
