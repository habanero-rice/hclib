#include "hcpp.h"
#include <sys/time.h>

using namespace std;

static int threshold = 10;

int fib_serial(int n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

void fib(int n, int* res)
{
    if (n <= threshold) {
        *res = fib_serial(n);
    }
    else {
	int x, y;
	hcpp::finish([n, &x, &y]( ) {
  	    hcpp::async([n, &x]( ){fib(n-1, &x);});
  	    fib(n-2, &y);
	});
	*res = x + y;
    }
}

long get_usecs (void)
{
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

int main (int argc, char ** argv) {
  hcpp::init(&argc, argv);
  int n = 40;
  int res;
  if(argc > 1) n = atoi(argv[1]);
  if(argc > 2) threshold = atoi(argv[2]);

  printf("Starting Fib(%d)..\n",n);
  long start = get_usecs();
  fib(n, &res);
  long end = get_usecs();
  double dur = ((double)(end-start))/1000000;
  printf("Fib(%d) = %d. Time = %f\n",n,res,dur);
  hcpp::finalize();
  return 0;
}

