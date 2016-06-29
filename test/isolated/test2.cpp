#include "hclib_cpp.h"

using namespace std;
#define LIMIT 100000

int main (int argc, char ** argv) {
  hclib::launch([&]() {
    int *ptr1 = new int;
    int *ptr2 = new int;
    int *ptr3 = new int;
    int *ptr4 = new int;
    *ptr1 = 0;
    *ptr2 = 0;
    *ptr3 = 0;
    *ptr4 = 0;

    hclib::enable_isolation(ptr1, ptr2, ptr3, ptr4);

    _loop_domain_t loop = {0, LIMIT, 1, 1};
    hclib::finish([&]() {
      hclib::forasync1D(&loop, [&](int i) {
        hclib::isolated(ptr1, ptr2, ptr3, ptr4, [=]() {
          *ptr1 += 1;
          *ptr2 += 1;
          *ptr3 += 1;
          *ptr4 += 1;
        });
      });
    });

    printf("Result=%d\n",*ptr1);
    assert(*ptr1==LIMIT && "Test Failed");
    assert(*ptr2==LIMIT && "Test Failed");
    assert(*ptr3==LIMIT && "Test Failed");
    assert(*ptr4==LIMIT && "Test Failed");
    printf("Test Passed\n");

    hclib::disable_isolation(ptr1, ptr2, ptr3, ptr4);
  });
  return 0;
}

