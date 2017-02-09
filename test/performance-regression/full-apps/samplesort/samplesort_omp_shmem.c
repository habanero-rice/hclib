#include "shmem.h"

/*********************************************************************

                    samplesort.c: source: http://www.cse.iitd.ernet.in/~dheerajb/MPI/codes/day-3/c/samplesort.c

     Objective      : To sort unsorted integers by sample sort algorithm 
                      Write a MPI program to sort n integers, using sample
                      sort algorithm on a p processor of PARAM 10000. 
                      Assume n is multiple of p. Sorting is defined as the
                      task of arranging an unordered collection of elements
                      into monotonically increasing (or decreasing) order. 

                      postcds: array[] is sorted in ascending order ANSI C 
                      provides a quicksort function called sorting(). Its 
                      function prototype is in the standard header file
                      <stdlib.h>

     Description    : 1. Partitioning of the input data and local sort :

                      The first step of sample sort is to partition the data.
                      Initially, each one of the p processors stores n/p
                      elements of the sequence of the elements to be sorted.
                      Let Ai be the sequence stored at processor Pi. In the
                      first phase each processor sorts the local n/p elements
                      using a serial sorting algorithm. (You can use C 
                      library sorting() for performing this local sort).

                      2. Choosing the Splitters : 

                      The second phase of the algorithm determines the p-1
                      splitter elements S. This is done as follows. Each 
                      processor Pi selects p-1 equally spaced elements from
                      the locally sorted sequence Ai. These p-1 elements
                      from these p(p-1) elements are selected to be the
                      splitters.

                      3. Completing the sort :

                      In the third phase, each processor Pi uses the splitters 
                      to partition the local sequence Ai into p subsequences
                      Ai,j such that for 0 <=j <p-1 all the elements in Ai,j
                      are smaller than Sj , and for j=p-1 (i.e., the last 
                      element) Ai, j contains the rest elements. Then each 
                      processor i sends the sub-sequence Ai,j to processor Pj.
                      Finally, each processor merge-sorts the received
                      sub-sequences, completing the sorting algorithm.

     Input          : Process with rank 0 generates unsorted integers 
                      using C library call rand().

     Output         : Process with rank 0 stores the sorted elements in 
                      the file sorted_data_out.

*********************************************************************/


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// #include <shmem.h>

#define SHMEM_SYNC_VALUE (-1L)
#define _SHMEM_SYNC_VALUE               SHMEM_SYNC_VALUE

#define SHMEM_INTERNAL_F2C_SCALE ( sizeof (long) / sizeof (int) )
#define SHMEM_BCAST_SYNC_SIZE           (128L / SHMEM_INTERNAL_F2C_SCALE)
#define _SHMEM_BCAST_SYNC_SIZE          SHMEM_BCAST_SYNC_SIZE

#define SIZE 100000
#define TYPE uint64_t
long pSync[_SHMEM_BCAST_SYNC_SIZE];
#define ASYNC_SHMEM
#define VERIFY
#define HC_GRANULARITY  0

static int compare(const void *i, const void *j)
{
  if ((*(TYPE*)i) > (*(TYPE *)j))
    return (1);
  if ((*(TYPE *)i) < (*(TYPE *)j))
    return (-1);
  return (0);
}

int partition(TYPE* data, int left, int right) {
  int i = left;
  int j = right;
  TYPE tmp;
  TYPE pivot = data[(left + right) / 2];
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

typedef struct sort_data_t {
  TYPE *buffer;
  int left;
  int right;
} sort_data_t;

void par_sort(void* arg) {
  sort_data_t *in = (sort_data_t*) arg;
  TYPE* data = in->buffer;
  int left = in->left; 
  int right = in->right;

  if (right - left + 1 > HC_GRANULARITY) {
    int index = partition(data, left, right);
#pragma omp parallel
    {
#pragma omp single nowait
        {
        if (left < index - 1) {
          sort_data_t* buf = (sort_data_t*) malloc(sizeof(sort_data_t)); 
          buf->buffer = data;
          buf->left = left;
          buf->right = index - 1; 
#pragma omp task
          {
              par_sort(buf);
          }
        }
        if (index < right) {
          sort_data_t* buf = (sort_data_t*) malloc(sizeof(sort_data_t)); 
          buf->buffer = data;
          buf->left = index;
          buf->right = right; 
#pragma omp task
          {
              par_sort(buf);
          }
        }
        }
    }
  }
  else {
    //  quicksort in C library
    qsort(data+left, right - left + 1, sizeof(TYPE), compare);
  }
  free(arg);
}

void sorting(TYPE* buffer, int size) {
  sort_data_t* buf = (sort_data_t*) malloc(sizeof(sort_data_t)); 
  buf->buffer = buffer;
  buf->left = 0;
  buf->right = size - 1; 
  par_sort(buf);
}

int main (int argc, char *argv[]) {
  /**** Initialising ****/
#pragma omp_to_hclib
  {
  shmem_init (); 
  /* Variable Declarations */

  int 	     Numprocs,MyRank, Root = 0;
  int 	     i,j,k, NoofElements, NoofElements_Bloc,
				  NoElementsToSort;
  int 	     count, temp;
  TYPE 	     *Input, *InputData;
  TYPE 	     *Splitter, *AllSplitter;
  TYPE 	     *Buckets, *BucketBuffer, *LocalBucket;
  TYPE 	     *OutputBuffer, *Output;
  
  MyRank = shmem_my_pe ();
  Numprocs = shmem_n_pes ();
  NoofElements = SIZE;

  if(( NoofElements % Numprocs) != 0){
    if(MyRank == Root)
      printf("Number of Elements are not divisible by Numprocs \n");
    shmem_finalize ();
    exit(0);
  }
  /**** Reading Input ****/
  
  Input = (TYPE *) shmem_malloc (NoofElements*sizeof(*Input));
  if(Input == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  if (MyRank == Root){
    /* Initialise random number generator  */ 
    printf ("Generating input Array for Sorting %d uint64_t numbers\n",SIZE);
    srand48((TYPE)NoofElements);
    for(i=0; i< NoofElements; i++) {
      Input[i] = rand();
    }
  }

  /**** Sending Data ****/

  NoofElements_Bloc = NoofElements / Numprocs;
  InputData = (TYPE *) shmem_malloc (NoofElements_Bloc * sizeof (*InputData));
  if(InputData == NULL) {
    printf("Error : Can not allocate memory \n");
  }
  //MPI_Scatter(Input, NoofElements_Bloc, TYPE_MPI, InputData, 
  //				  NoofElements_Bloc, TYPE_MPI, Root, MPI_COMM_WORLD);

  shmem_barrier_all();
  if(MyRank == Root) {
    for(i=0; i<Numprocs; i++) {
      TYPE* start = &Input[i * NoofElements_Bloc];
      shmem_put64(InputData, start, NoofElements_Bloc, i);
    }
  }
  shmem_barrier_all();

  /**** Sorting Locally ****/
  sorting(InputData, NoofElements_Bloc);

  /**** Choosing Local Splitters ****/
  Splitter = (TYPE *) shmem_malloc (sizeof (TYPE) * (Numprocs-1));
  if(Splitter == NULL) {
    printf("Error : Can not allocate memory \n");
  }
  for (i=0; i< (Numprocs-1); i++){
        Splitter[i] = InputData[NoofElements/(Numprocs*Numprocs) * (i+1)];
  } 

  /**** Gathering Local Splitters at Root ****/
  AllSplitter = (TYPE *) shmem_malloc (sizeof (TYPE) * Numprocs * (Numprocs-1));
  if(AllSplitter == NULL) {
    printf("Error : Can not allocate memory \n");
  }
  //MPI_Gather (Splitter, Numprocs-1, TYPE_MPI, AllSplitter, Numprocs-1, 
  //				  TYPE_MPI, Root, MPI_COMM_WORLD);
  shmem_barrier_all();
  TYPE* target_index = &AllSplitter[MyRank * (Numprocs-1)];
  shmem_put64(target_index, Splitter, Numprocs-1, Root);
  shmem_barrier_all();

  /**** Choosing Global Splitters ****/
  if (MyRank == Root){
    sorting (AllSplitter, Numprocs*(Numprocs-1));

    for (i=0; i<Numprocs-1; i++)
      Splitter[i] = AllSplitter[(Numprocs-1)*(i+1)];
  }
  
  /**** Broadcasting Global Splitters ****/
  //MPI_Bcast (Splitter, Numprocs-1, TYPE_MPI, 0, MPI_COMM_WORLD);
  { int _i; for(_i=0; _i<_SHMEM_BCAST_SYNC_SIZE; _i++) { pSync[_i] = _SHMEM_SYNC_VALUE; } shmem_barrier_all(); }
  shmem_broadcast64(Splitter, Splitter, Numprocs-1, 0, 0, 0, Numprocs, pSync);
  shmem_barrier_all();

  /**** Creating Numprocs Buckets locally ****/
  Buckets = (TYPE *) shmem_malloc (sizeof (TYPE) * (NoofElements + Numprocs));  
  if(Buckets == NULL) {
    printf("Error : Can not allocate memory \n");
  }
  
  j = 0;
  k = 1;

  for (i=0; i<NoofElements_Bloc; i++){
    if(j < (Numprocs-1)){
       if (InputData[i] < Splitter[j]) 
			 Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i]; 
       else{
	       Buckets[(NoofElements_Bloc + 1) * j] = k-1;
		    k=1;
			 j++;
		    i--;
       }
    }
    else 
       Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i];
  }
  Buckets[(NoofElements_Bloc + 1) * j] = k - 1;
  shmem_free(Splitter);
  shmem_free(AllSplitter);
      
  /**** Sending buckets to respective processors ****/

  BucketBuffer = (TYPE *) shmem_malloc (sizeof (TYPE) * (NoofElements + Numprocs));
  if(BucketBuffer == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  //MPI_Alltoall (Buckets, NoofElements_Bloc + 1, TYPE_MPI, BucketBuffer, 
  //					 NoofElements_Bloc + 1, TYPE_MPI, MPI_COMM_WORLD);
  shmem_barrier_all();
  for(i=0; i<Numprocs; i++) {
    shmem_put64(&BucketBuffer[MyRank*(NoofElements_Bloc + 1)], &Buckets[i*(NoofElements_Bloc + 1)],  NoofElements_Bloc + 1, i);   
  }
  shmem_barrier_all();

  /**** Rearranging BucketBuffer ****/
  LocalBucket = (TYPE *) shmem_malloc (sizeof (TYPE) * 2 * NoofElements / Numprocs);
  if(LocalBucket == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  count = 1;

  for (j=0; j<Numprocs; j++) {
  k = 1;
    for (i=0; i<BucketBuffer[(NoofElements/Numprocs + 1) * j]; i++) 
      LocalBucket[count++] = BucketBuffer[(NoofElements/Numprocs + 1) * j + k++];
  }
  LocalBucket[0] = count-1;
    
  /**** Sorting Local Buckets using Bubble Sort ****/
  /*sorting (InputData, NoofElements_Bloc, sizeof(int), intcompare); */

  NoElementsToSort = LocalBucket[0];
  sorting (&LocalBucket[1], NoElementsToSort); 

  /**** Gathering sorted sub blocks at root ****/
  OutputBuffer = (TYPE *) shmem_malloc (sizeof(TYPE) * 2 * NoofElements);
  if(OutputBuffer == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  //MPI_Gather (LocalBucket, 2*NoofElements_Bloc, TYPE_MPI, OutputBuffer, 
  //				  2*NoofElements_Bloc, TYPE_MPI, Root, MPI_COMM_WORLD);
  shmem_barrier_all();
  target_index = &OutputBuffer[MyRank * (2*NoofElements_Bloc)];
  shmem_put64(target_index, LocalBucket, 2*NoofElements_Bloc, Root);
  shmem_barrier_all();

  /**** Rearranging output buffer ****/
  if (MyRank == Root){
    Output = (TYPE *) malloc (sizeof (TYPE) * NoofElements);
    count = 0;
    for(j=0; j<Numprocs; j++){
      k = 1;
      for(i=0; i<OutputBuffer[(2 * NoofElements/Numprocs) * j]; i++) 
        Output[count++] = OutputBuffer[(2*NoofElements/Numprocs) * j + k++];
      }
       printf ( "Number of Elements to be sorted : %d \n", NoofElements);
       TYPE prev = 0;
       int fail = 0;
       for (i=0; i<NoofElements; i++){
         if(Output[i] < prev) { printf("Failed at index %d\n",i); fail = 1; }
         prev = Output[i];
       }
       if(fail) printf("Sorting FAILED\n");  
       else  printf("Sorting PASSED\n");
  	free(Output);
  }/* MyRank==0*/

  shmem_free(Input);
  shmem_free(OutputBuffer);
  shmem_free(InputData);
  shmem_free(Buckets);
  shmem_free(BucketBuffer);
  shmem_free(LocalBucket);

   /**** Finalize ****/
  shmem_finalize();
  }
}

