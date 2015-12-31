#include "hclib_cpp.h"
#include <iostream>
using namespace std;

static int threshold = 2;

int fib_serial(int n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

void fib(int n, hclib::promise_t* res) {
  int* r = new int;
  if (n <= threshold) {
    *r = fib_serial(n);
    hclib::promise_put(res, r);
    return;
  } 

  // compute f1 asynchronously
  hclib::promise_t* f1 = hclib::promise_create();
  hclib::async([=]() { 
    fib(n - 1, f1);
  });

  // compute f2 serially (f1 is done asynchronously).
  hclib::promise_t* f2 = hclib::promise_create();
  hclib::async([=]() { 
    fib(n - 2, f2);
  });

  // wait for dependences, before updating the result
  hclib::asyncAwait(f1, f2, [=]() {
    *r = *((int*) hclib::promise_get(f1)) + *((int*) hclib::promise_get(f2));
    hclib::promise_put(res, r);
  });
}

int main(int argc, char** argv) {
    hclib::launch(&argc, argv, [&]() {
        int n = argc == 1 ? 30 : atoi(argv[1]);
        threshold = argc == 2 ? 10 : atoi(argv[2]);
        hclib::promise_t* promise = hclib::promise_create();
        hclib::start_finish();
        fib(n, promise);
        hclib::end_finish();
        int res = *((int*)hclib::promise_get(promise));
        cout << "Fib(" << n << ") = " << res << endl;
    });
    return 0;
}
