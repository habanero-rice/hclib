#include"hclib_cpp.h"
#include<cmath>

/*
 * iterative averaging using forasync1D
 */

#define VERIFY /*comment this line to turn of computation validation*/
//48 * 256 * 2048
#define SIZE 25165824
#define ITERATIONS 64

double* myNew, *myVal, *initialOutput;
int n;

int ceilDiv(int d) {
  int m = SIZE / d;
  if (m * d == SIZE) {
    return m;
  } else {
    return (m + 1);
  }
}

/* Validate the computatoin */
void validateOutput() {
  for (int i = 0; i < SIZE + 2; i++) {
    double init = initialOutput[i];
    double curr = myVal[i];
    double diff = std::abs(init - curr);
    if (diff > 1e-20) {
      printf("ERROR: validation failed!\n");
      printf("Diff: myVal[%d]=%.3f != initialOutput[%d]=%.3f",i,curr,i,init);
      break;
    }
  }
}

/* Sequential invokation of computation */
void runSequential() {
  for (int iter = 0; iter < ITERATIONS; iter++) {
    for(int j=1; j<=SIZE; j++) {
        myNew[j] = (initialOutput[j - 1] + initialOutput[j + 1]) / 2.0;
    }
    double* temp = myNew;
    myNew = initialOutput;
    initialOutput = temp;
  }
}

/* Parallel invocation of computation */
void runParallel() {
  /* note we are not using ceil(...) as we did in async-finish version.
   * If there are some remainder iterations after division then forasync1D
   * will take care of it
   * Also, note that we have highBound=SIZE+1 (matching the runSequential) 
   */
  hclib::loop_domain_1d* loop = new hclib::loop_domain_1d(1, SIZE+1);
  for (int iter = 0; iter < ITERATIONS; iter++) {
    hclib::finish([=]() {
      hclib::forasync1D(loop, [=](int j) {
        myNew[j] = (myVal[j - 1] + myVal[j + 1]) / 2.0;
      }, false, FORASYNC_MODE_RECURSIVE);
    });
    double* temp = myNew;
    myNew = myVal;
    myVal = temp;
  }
  delete loop;
}

int main(int argc, char** argv) {
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    myNew = new double[(SIZE + 2)];
    myVal = new double[(SIZE + 2)];
    initialOutput = new double[(SIZE + 2)];
    memset(myNew, 0, sizeof(double) * (SIZE + 2));
    memset(myVal, 0, sizeof(double) * (SIZE + 2));
    memset(initialOutput, 0, sizeof(double) * (SIZE + 2));
    initialOutput[SIZE + 1] = 1.0;
    myVal[SIZE + 1] = 1.0;
#ifdef VERIFY
    long start_seq = hclib_current_time_ms();
    runSequential();
    long end_seq = hclib_current_time_ms();
    double dur_seq = ((double)(end_seq-start_seq))/1000;
    printf("Sequential Time = %.3f\n",dur_seq);
#endif
    long start = hclib_current_time_ms();
    runParallel();
    long end = hclib_current_time_ms();
    double dur = ((double)(end-start))/1000;
#ifdef VERIFY
    validateOutput();
#endif
    printf("Time = %.3f\n",dur);
    delete(myNew);
    delete(myVal);
    delete(initialOutput);
  });
}

