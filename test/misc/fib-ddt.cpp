#include "hclib_cpp.h"
#include <iostream>
using namespace std;

static int threshold = 2;

int fib_serial(int n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

void fib(int n, hclib::ddf_t* res) {
  int* r = new int;
  if (n <= threshold) {
    *r = fib_serial(n);
    hclib::ddf_put(res, r);
    return;
  } 

  // compute f1 asynchronously
  hclib::ddf_t* f1 = hclib::ddf_create();
  hclib::async([=]() { 
    fib(n - 1, f1);
  });

  // compute f2 serially (f1 is done asynchronously).
  hclib::ddf_t* f2 = hclib::ddf_create();
  hclib::async([=]() { 
    fib(n - 2, f2);
  });

  // wait for dependences, before updating the result
  hclib::asyncAwait(f1, f2, [=]() {
    *r = *((int*) hclib::ddf_get(f1)) + *((int*) hclib::ddf_get(f2));
    hclib::ddf_put(res, r);
  });
}

int main(int argc, char** argv) {
  hclib::init(&argc, argv);
  int n = argc == 1 ? 30 : atoi(argv[1]);
  threshold = argc == 2 ? 10 : atoi(argv[2]);
  hclib::ddf_t* ddf = hclib::ddf_create();
  hclib::start_finish();
  fib(n, ddf);
  hclib::end_finish();
  int res = *((int*)hclib::ddf_get(ddf));
  cout << "Fib(" << n << ") = " << res << endl;
  hclib::finalize();
  return 0;
}
