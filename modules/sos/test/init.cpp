#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };
    hclib::launch(deps, 2, [] {
        std::cout << "Hello world from rank " << hclib::shmem_my_pe() << std::endl;
    });
    return 0;
}
