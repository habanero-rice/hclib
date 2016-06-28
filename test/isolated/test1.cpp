#include "hclib_cpp.h"

using namespace std;
#define LIMIT 1000

int main (int argc, char ** argv) {
  hclib::launch([&]() {
    int *ptr = new int;
    *ptr = 0;

    hclib::enable_isolation(ptr);

    hclib::finish([=]() {
      for(int i=0; i<LIMIT; i++) {
        hclib::async([=]() {
          hclib::isolated(ptr, [=]() {
            *ptr += 1;
          });
        });
      }
    });

    printf("Result=%d\n",*ptr);
    assert(*ptr==LIMIT && "Test Failed");
    printf("Test Passed\n");
    hclib::disable_isolation(ptr);
  });
  return 0;
}

