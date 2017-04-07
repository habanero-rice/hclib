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
 * HC CONCORD foreach add2d.hc example 
 */

#include "hclib_cpp.h"

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
        int num_iters1;
        int num_iters2;
        int tilesize1;
        int tilesize2;
        int i;
        int *a,*b,*c;

        if (argc != 5) {
            printf("USAGE:./arrayadd2d NUM_ITERS1 NUM_ITERS2 TILE_SIZE1 TILE_SIZE2\n");
            exit(1);
        }
        num_iters1 = atoi(argv[1]);
        num_iters2 = atoi(argv[2]);
        tilesize1 = atoi(argv[3]);
        tilesize2 = atoi(argv[4]);

        a=(int*)malloc(sizeof(int)*num_iters1*num_iters2);
        b=(int*)malloc(sizeof(int)*num_iters1*num_iters2);
        c=(int*)malloc(sizeof(int)*num_iters1*num_iters2);

        //Initialize the Values
        for(i=0; i<num_iters1*num_iters2; i++){

            a[i]=0;
            b[i]=100;
            c[i]=1;

        }
        //Add the elements of arrays b and c and store them in a
        hclib::finish([=]() {
            hclib::loop_domain_t loop[2] = {{0, num_iters1, 1, tilesize1}, {0, num_iters2, 1, tilesize2}} ;
            hclib::forasync2D(loop, [=](int i, int j) {
                   a[i*num_iters2+j]=b[i*num_iters2+j]+c[i*num_iters2+j];
            }, FORASYNC_MODE_RECURSIVE);
            //}, FORASYNC_MODE_FLAT);
        });

        check(a,101,num_iters1*num_iters2);
        printf("Test passed\n");
    });
	return 0;
}
