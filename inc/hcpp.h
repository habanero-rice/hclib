#include "hcpp-utils.h"

extern int _hcpp_main(int argc, char **argv);

int main(int argc, char **argv)
{
  hclib_init(&argc, argv);

  // start the user main function
  int rv = _hcpp_main(argc, argv);

  hclib_finalize();
  return rv;
}

#define main _hcpp_main

