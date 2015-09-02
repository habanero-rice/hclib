/*
 * HC CONCORD foreach add2d.hc example 
 */

#include "hcpp.h"

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
	hcpp::init(&argc, argv);
	int num_iters1;
	int num_iters2;
	int tilesize1;
	int tilesize2;
	int i,j;
	int *a,*b,*c;

	if(argc!=5){printf("USAGE:./arrayadd2d NUM_ITERS1 NUM_ITERS2 TILE_SIZE1 TILE_SIZE2\n");return -1;}
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
	hcpp::start_finish();
        hcpp::_loop_domain_t loop[2] = {{0, num_iters1, 1, tilesize1}, {0, num_iters2, 1, tilesize2}} ;
        hcpp::forasync2D(loop, [=](int i, int j) {
               a[i*num_iters2+j]=b[i*num_iters2+j]+c[i*num_iters2+j];
        }, FORASYNC_MODE_RECURSIVE);
        //}, FORASYNC_MODE_FLAT);
        hcpp::end_finish();

	check(a,101,num_iters1*num_iters2);
	printf("Test passed\n");
	hcpp::finalize();
	return 0;
}
