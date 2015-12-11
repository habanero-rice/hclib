#include "hclib_cpp.h"
#include <sys/time.h>

using namespace std;

static int threshold = 10;

int fib_serial(int n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

int fib(int n)
{
    if (n <= threshold) {
        return fib_serial(n);
    }
    else {
	int x, y;
	hclib::finish([n, &x, &y]( ) {
  	    hclib::async([n, &x]( ){x = fib(n-1);});
  	    y = fib(n-2);
	});
	return x + y;
    }
}

long get_usecs (void)
{
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

int main (int argc, char ** argv) {
  hclib::launch(&argc, argv, [&]() {
      int n = 40;
      if(argc > 1) n = atoi(argv[1]);
      if(argc > 2) threshold = atoi(argv[2]);

      printf("Starting Fib(%d)..\n",n);
      long start = get_usecs();
      int res = fib(n);
      long end = get_usecs();
      double dur = ((double)(end-start))/1000000;
      printf("Fib(%d) = %d. Time = %f\n",n,res,dur);
  });
  return 0;
}

