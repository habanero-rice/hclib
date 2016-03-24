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
    res->put(r);
    return;
  } 

  // compute f1 asynchronously
  hclib::promise_t* f1 = new hclib::promise_t();
  hclib::async([=]() { 
    fib(n - 1, f1);
  });

  // compute f2 serially (f1 is done asynchronously).
  hclib::promise_t* f2 = new hclib::promise_t();
  hclib::async([=]() { 
    fib(n - 2, f2);
  });

  // wait for dependences, before updating the result
  hclib::async_await([=] {
    *r = *((int*) f1->get_future()->get()) + *((int*) f2->get_future()->get());
    res->put(r);
  }, f1->get_future(), f2->get_future());
}

int main(int argc, char** argv) {
    hclib::launch(&argc, argv, [&]() {
        int n = argc == 1 ? 30 : atoi(argv[1]);
        threshold = argc == 2 ? 10 : atoi(argv[2]);
        hclib::promise_t* promise = new hclib::promise_t();
        hclib::finish([=] {
            fib(n, promise);
        });
        int res = *((int*)promise->get_future()->get());
        cout << "Fib(" << n << ") = " << res << endl;
    });
    return 0;
}
