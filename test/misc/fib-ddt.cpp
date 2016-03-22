#include "hcpp.h"
using namespace std;

static int threshold = 2;

int fib_serial(int n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

void fib(int n, hcpp::DDF_t* res) {
  int* r = new int;
  if (n <= threshold) {
    *r = fib_serial(n);
    hcpp::ddf_put(res, r);
    return;
  } 

  // compute f1 asynchronously
  hcpp::DDF_t* f1 = hcpp::ddf_create();
  hcpp::async([=]() { 
    fib(n - 1, f1);
  });

  // compute f2 serially (f1 is done asynchronously).
  hcpp::DDF_t* f2 = hcpp::ddf_create();
  hcpp::async([=]() { 
    fib(n - 2, f2);
  });

  // wait for dependences, before updating the result
  hcpp::asyncAwait(f1, f2, [=]() {
    *r = *((int*) hcpp::ddf_get(f1)) + *((int*) hcpp::ddf_get(f2));
    hcpp::ddf_put(res, r);
  });
}

int main(int argc, char** argv) {
  hcpp::init(&argc, argv);
  int n = argc == 1 ? 30 : atoi(argv[1]);
  threshold = argc == 2 ? 10 : atoi(argv[2]);
  hcpp::DDF_t* ddf = hcpp::ddf_create();
  hcpp::start_finish();
  fib(n, ddf);
  hcpp::end_finish();
  int res = *((int*)ddf_get(ddf));
  cout << "Fib(" << n << ") = " << res << endl;
  hcpp::finalize();
  return 0;
}
