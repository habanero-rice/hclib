
/*
********************************************************************

                    Example 28 (samplesort.c)

     Objective      : To sort unsorted integers by sample sort algorithm 
                      Write a MPI program to sort n integers, using sample
                      sort algorithm on a p processor of PARAM 10000. 
                      Assume n is multiple of p. Sorting is defined as the
                      task of arranging an unordered collection of elements
                      into monotonically increasing (or decreasing) order. 

                      postcds: array[] is sorted in ascending order ANSI C 
                      provides a quicksort function called qsort(). Its 
                      function prototype is in the standard header file
                      <stdlib.h>

     Description    : 1. Partitioning of the input data and local sort :

                      The first step of sample sort is to partition the data.
                      Initially, each one of the p processors stores n/p
                      elements of the sequence of the elements to be sorted.
                      Let Ai be the sequence stored at processor Pi. In the
                      first phase each processor sorts the local n/p elements
                      using a serial sorting algorithm. (You can use C 
                      library qsort() for performing this local sort).

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

********************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

#define SIZE 10

/**** Function Declaration Section ****/

static int intcompare(const void *i, const void *j)
{
  if ((*(int *)i) > (*(int *)j))
    return (1);
  if ((*(int *)i) < (*(int *)j))
    return (-1);
  return (0);
}



main (int argc, char *argv[])
{
  /* Variable Declarations */

  int 	     Numprocs,MyRank, Root = 0;
  int 	     i,j,k, NoofElements, NoofElements_Bloc,
				  NoElementsToSort;
  int 	     count, temp;
  int 	     *Input, *InputData;
  int 	     *Splitter, *AllSplitter;
  int 	     *Buckets, *BucketBuffer, *LocalBucket;
  int 	     *OutputBuffer, *Output;
  FILE 	     *InputFile, *fp;
  MPI_Status  status; 
  
  /**** Initialising ****/
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

  if(argc != 2) {
      if(MyRank ==0) printf(" Usage : run size\n");
	 MPI_Finalize();
	 exit(0);
	}

  /**** Reading Input ****/
  
  if (MyRank == Root){

    NoofElements = atoi(argv[1]);
    Input = (int *) malloc (NoofElements*sizeof(int));
	 if(Input == NULL) {
		printf("Error : Can not allocate memory \n");
    }

  /* Initialise random number generator  */ 
  // printf ( "Input Array for Sorting \n\n ");
  //   srand48((unsigned int)NoofElements);
  //    for(i=0; i< NoofElements; i++) {
  //      Input[i] = rand();
  //      printf ("%d   ",Input[i]);
  //   }
  }
  // printf ( "\n\n ");

  /**** Sending Data ****/
  MPI_Bcast (&NoofElements, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(( NoofElements % Numprocs) != 0){
	    if(MyRank == Root)
		printf("Number of Elements are not divisible by Numprocs \n");
            MPI_Finalize();
	    exit(0);
  }

  NoofElements_Bloc = NoofElements / Numprocs;
  InputData = (int *) malloc (NoofElements_Bloc * sizeof (int));

  MPI_Scatter(Input, NoofElements_Bloc, MPI_INT, InputData, 
				  NoofElements_Bloc, MPI_INT, Root, MPI_COMM_WORLD);

  /**** Sorting Locally ****/
  qsort ((char *) InputData, NoofElements_Bloc, sizeof(int), intcompare);

  /**** Choosing Local Splitters ****/
  Splitter = (int *) malloc (sizeof (int) * (Numprocs-1));
  for (i=0; i< (Numprocs-1); i++){
        Splitter[i] = InputData[NoofElements/(Numprocs*Numprocs) * (i+1)];
  } 

  /**** Gathering Local Splitters at Root ****/
  AllSplitter = (int *) malloc (sizeof (int) * Numprocs * (Numprocs-1));
  MPI_Gather (Splitter, Numprocs-1, MPI_INT, AllSplitter, Numprocs-1, 
				  MPI_INT, Root, MPI_COMM_WORLD);

  /**** Choosing Global Splitters ****/
  if (MyRank == Root){
    qsort ((char *) AllSplitter, Numprocs*(Numprocs-1), sizeof(int), intcompare);

    for (i=0; i<Numprocs-1; i++)
      Splitter[i] = AllSplitter[(Numprocs-1)*(i+1)];
  }
  
  /**** Broadcasting Global Splitters ****/
  MPI_Bcast (Splitter, Numprocs-1, MPI_INT, 0, MPI_COMM_WORLD);

  /**** Creating Numprocs Buckets locally ****/
  Buckets = (int *) malloc (sizeof (int) * (NoofElements + Numprocs));
  
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
      
  /**** Sending buckets to respective processors ****/

  BucketBuffer = (int *) malloc (sizeof (int) * (NoofElements + Numprocs));

  MPI_Alltoall (Buckets, NoofElements_Bloc + 1, MPI_INT, BucketBuffer, 
					 NoofElements_Bloc + 1, MPI_INT, MPI_COMM_WORLD);

  /**** Rearranging BucketBuffer ****/
  LocalBucket = (int *) malloc (sizeof (int) * 2 * NoofElements / Numprocs);

  count = 1;

  for (j=0; j<Numprocs; j++) {
  k = 1;
    for (i=0; i<BucketBuffer[(NoofElements/Numprocs + 1) * j]; i++) 
      LocalBucket[count++] = BucketBuffer[(NoofElements/Numprocs + 1) * j + k++];
  }
  LocalBucket[0] = count-1;
    
  /**** Sorting Local Buckets using Bubble Sort ****/
  /*qsort ((char *) InputData, NoofElements_Bloc, sizeof(int), intcompare); */

  NoElementsToSort = LocalBucket[0];
  qsort ((char *) &LocalBucket[1], NoElementsToSort, sizeof(int), intcompare); 

  /**** Gathering sorted sub blocks at root ****/
  if(MyRank == Root) {
  		OutputBuffer = (int *) malloc (sizeof(int) * 2 * NoofElements);
  		Output = (int *) malloc (sizeof (int) * NoofElements);
  }

  MPI_Gather (LocalBucket, 2*NoofElements_Bloc, MPI_INT, OutputBuffer, 
				  2*NoofElements_Bloc, MPI_INT, Root, MPI_COMM_WORLD);

  /**** Rearranging output buffer ****/
	if (MyRank == Root){
		count = 0;
		for(j=0; j<Numprocs; j++){
          k = 1;
      	 for(i=0; i<OutputBuffer[(2 * NoofElements/Numprocs) * j]; i++) 
				 Output[count++] = OutputBuffer[(2*NoofElements/Numprocs) * j + k++];
    	}

      /**** Printng the output ****/
//     	if ((fp = fopen("sort.out", "w")) == NULL){
//          	printf("Can't Open Output File \n");
//       		exit(0);
//     	}
// 		 
//     	fprintf (fp, "Number of Elements to be sorted : %d \n", NoofElements);
//     	printf ( "Number of Elements to be sorted : %d \n", NoofElements);
//     	fprintf (fp, "The sorted sequence is : \n");
// 	printf( "Sorted output sequence is\n\n");
//     	for (i=0; i<NoofElements; i++){
// 	      	fprintf(fp, "%d\n", Output[i]);
// 	      	printf( "%d   ", Output[i]);
// 	}
// 	printf ( " \n " );
// 	fclose(fp);
    	free(Input);
  	free(OutputBuffer);
  	free(Output);
   }/* MyRank==0*/

  	free(InputData);
  	free(Splitter);
  	free(AllSplitter);
  	free(Buckets);
  	free(BucketBuffer);
  	free(LocalBucket);

   /**** Finalize ****/
   MPI_Finalize();
}



