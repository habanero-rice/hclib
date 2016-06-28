#include "hclib_cpp.h"

using namespace std;

int main (int argc, char ** argv) {
  hclib::launch([&]() {
    double *ptr = new double;

    hclib::enable_isolation(ptr);

    hclib::isolated(ptr, [=]() {
      // Do something..
    });

    hclib::disable_isolation(ptr);
  });
  return 0;
}

