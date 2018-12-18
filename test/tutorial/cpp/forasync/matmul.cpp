#include "hclib_cpp.h"
#include <sys/time.h>

double **a, **b, **c;

/*
 * Timer routine
 */
long get_usecs (void) {
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

/*
 * Intialize input matrices and the output matrix
 */
void init(int n) {
  a = new double*[n];
  b = new double*[n];
  c = new double*[n];

  // Demonstration of forasync1D
  hclib::loop_domain_1d* loop1 = new hclib::loop_domain_1d(n);
  hclib::finish([&]() {
    hclib::forasync1D(loop1, [=](int i) {
      a[i] = new double[n];
      b[i] = new double[n];
      c[i] = new double[n];
    }, false, FORASYNC_MODE_RECURSIVE);
  });
  delete loop1;
  
  // Demonstration of forasync2D
  hclib::loop_domain_2d* loop2 = new hclib::loop_domain_2d(n, n);
  hclib::finish([&]() {
    hclib::forasync2D(loop2, [=](int i, int j) {
      a[i][j] = 1.0;  
      b[i][j] = 1.0;  
      c[i][j] = 0;
    }, false, FORASYNC_MODE_RECURSIVE);    
  });
  delete loop2;
}

/*
 * release memory
 */
void freeall(int n) {
  for(int i=0; i<n; i++) {
    delete(a[i]);
    delete(b[i]);
    delete(c[i]);
  }
  delete (a);
  delete (b);
  delete (c);
}

/*
 * Validate the result of matrix multiplication
 */
int verify(int n) {
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      if(c[i][j] != n) {
        printf("result = %.3f\n",c[i][j]);
        return false;
      }
    }
  }
  return true;
}

void multiply(int n) {
  // Demonstration of forasync2D
  hclib::loop_domain_2d* loop = new hclib::loop_domain_2d(n, n);
  hclib::finish([=]() {
    hclib::forasync2D(loop, [=](int i, int j) {
      for(int k=0; k<n; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }, false, FORASYNC_MODE_RECURSIVE);    
  });
  delete loop;
}

int main(int argc, char** argv) {
  int n = argc>1 ? atoi(argv[1]) : 1024;
  printf("Size = %d\n",n);
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    // initialize
    init(n);
    //start timer
    long start = get_usecs();
    //multiply matrices
    multiply(n);
    //end timer
    long end = get_usecs();
    double dur = ((double)(end-start))/1000000;
    //validate result
    int result = verify(n);
    printf("MatrixMultiplication result = %d, Time = %.3f\n",result, dur);
    //release memory
    freeall(n);
  });
  return 0;
}
