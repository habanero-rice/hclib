/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
