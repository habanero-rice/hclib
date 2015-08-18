#include "hcpp.h"
#include <sys/time.h>

using namespace std;

void fib(int n, int* res)
{
    if (n <= 2) {
        *res = 1;
    }
    else {
	int* x = new int, *y = new int;
    	if(n<=10) {
	  fib(n-1, x);
          fib(n-2, y);
	}
	else {
	  hcpp::start_finish();
  	  hcpp::async([=](){fib(n-1, x);});
  	  hcpp::async([=](){fib(n-2, y);});
	  hcpp::end_finish();
	}
	*res = *x + *y;
	free(x);
	free(y);
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
  int n = 30;
  int* res = new int;
  if(argc > 1) n = atoi(argv[1]);

  printf("Starting Fib(%d)..\n",n);
  long start = get_usecs();
  fib(n, res);
  long end = get_usecs();
  double dur = ((double)(end-start))/1000000;
  printf("Fib(%d) = %d. Time = %f\n",n,*res,dur);
  free(res);
  hcpp::finalize();
  return 0;
}

