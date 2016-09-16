#include "hclib_cpp.h"
#include "hclib_upcxx.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [argc, argv] {
        std::cerr << "Hello from UPC++ rank " << hclib::upcxx::myrank() <<
            " out of " << hclib::upcxx::ranks() << " rank(s)" << std::endl;
    });
    return 0;
}
