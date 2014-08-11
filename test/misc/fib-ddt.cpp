#include "hcpp.h"
using namespace std;
using namespace hcpp;

void fib(int n, DDF_t* res) {
  int* r = new int;
  if (n <= 0) {
    *r = 0;
    DDF_PUT(res, r);
    return;
  } else if (n == 1) {
    *r = 1;
    DDF_PUT(res, r);
    return;
  }

  // compute f1 asynchronously
  DDF_t* f1 = DDF_CREATE();
  async([=]() { 
    fib(n - 1, f1);
  });

  // compute f2 serially (f1 is done asynchronously).
  DDF_t* f2 = DDF_CREATE();
  fib(n - 2, f2);

  // wait for dependences, before updating the result
  asyncAwait(f1, f2, [=]() {
    *r = *((int*) DDF_GET(f1)) + *((int*) DDF_GET(f2));
    DDF_PUT(res, r);
  });
}

int main(int argc, char** argv) {
  int n = argc == 1 ? 20 : atoi(argv[1]);
  DDF_t* ddf = DDF_CREATE();
  hcpp::finish([=]() {
    fib(n, ddf);
  });
  int res = *((int*)DDF_GET(ddf));
  cout << "Fib(" << n << ") = " << res << endl;
  return 0;
}
