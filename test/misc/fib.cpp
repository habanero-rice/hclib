#include "hcpp.h"
#include <sys/time.h>

using namespace std;

void fib(int n, int* res)
{
    if (n <= 2) {
        *res = 1;
    }
    else {
	int x, y;
    	if(n<=20) {
	  fib(n-1, &x);
          fib(n-2, &y);
	}
	else {
	  hcpp::finish( [&] () {
  	    hcpp::async([&](){fib(n-1, &x);});
  	    hcpp::async([&](){fib(n-2, &y);});
	  });
	}
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
  int n = 40;
  int res;
  if(argc > 1) n = atoi(argv[1]);

  printf("Starting Fib(%d)..\n",n);
  long start = get_usecs();
  fib(n, &res);
  long end = get_usecs();
  double dur = ((double)(end-start))/1000000;
  printf("Fib(%d) = %d. Time = %f\n",n,res,dur);
  return 0;
}

