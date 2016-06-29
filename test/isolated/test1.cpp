#include "hclib_cpp.h"

using namespace std;
#define LIMIT 100000

int main (int argc, char ** argv) {
  hclib::launch([&]() {
    int *ptr = new int;
    *ptr = 0;

    hclib::enable_isolation(ptr);

    _loop_domain_t loop = {0, LIMIT, 1, 1};
    hclib::finish([&]() {
      hclib::forasync1D(&loop, [=](int i) {
        hclib::isolated(ptr, [=]() {
          *ptr += 1;
        });
      });
    });

    printf("Result=%d\n",*ptr);
    assert(*ptr==LIMIT && "Test Failed");
    printf("Test Passed\n");
    hclib::disable_isolation(ptr);
  });
  return 0;
}

