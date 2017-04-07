/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
	HCLIB_FINISH {
  	    hclib::async([n, &x]( ){x = fib(n-1);});
  	    y = fib(n-2);
	}
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
  hclib::launch([=]() {
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

