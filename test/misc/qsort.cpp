#include "hcpp.h"
#include <sys/time.h>

using namespace std;

#define ELEMENT_T uint64_t

int partition(ELEMENT_T* data, int left, int right) {
	int i = left;
	int j = right;
	ELEMENT_T tmp;
	ELEMENT_T pivot = data[(left + right) / 2];

	while (i <= j) {
		while (data[i] < pivot) i++;
		while (data[j] > pivot) j--;
		if (i <= j) {
			tmp = data[i];
			data[i] = data[j];
		data[j] = tmp;
			i++;
			j--;
		}
	}

	return i;
}

int compare(const void * a, const void * b)
{
	if ( *(ELEMENT_T*)a <  *(ELEMENT_T*)b ) return -1;
	else if ( *(ELEMENT_T*)a == *(ELEMENT_T*)b ) return 0;
	else return 1;
}

void sort(ELEMENT_T* data, int left, int right, ELEMENT_T threshold) {
	if (right - left + 1 > threshold) {
		int index = partition(data, left, right);
		hcpp::start_finish();
			if (left < index - 1) {
				hcpp::async([=]() {
                        		sort(data, left, (index - 1), threshold);
				});
			}

			if (index < right) {
				hcpp::async([=]() {
					sort(data, index, right, threshold);
				});
			}
		hcpp::end_finish();
	}
	else {
		//  quicksort in C++ library
		qsort(data+left, right - left + 1, sizeof(ELEMENT_T), compare);
	}

}

long get_usecs (void)
{
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

int main(int argc, char **argv) {
	hcpp::init(&argc, argv);
	int N = argc>1 ? atoi(argv[1]) : 1000000; // 1 million
        int threshold = argc>2 ? atoi(argv[2]) : (int)(0.01*N);
	printf("Sorting %d size array with threshold of %d\n",N,threshold);
  	ELEMENT_T* data = new ELEMENT_T[N];

	srand(1);
	for(int i=0; i<N; i++) {
		data[i] = (ELEMENT_T)rand();
	}	
	
	long start = get_usecs();
	sort(data, 0, N-1, threshold);
  	long end = get_usecs();
  	double dur = ((double)(end-start))/1000000;
	
	ELEMENT_T a =0, b;
	bool ok= true;
        for (int k=0; k<N; k++) {
        	b = data[k];
                ok &= (a <= b);
                a = b;
        }
        if(ok){
               printf("QuickSort passed, Time = %f\n",dur);
        }
        else{
               printf("QuickSort failed, Time = %f\n",dur);
        }
	hcpp::finalize();
}
