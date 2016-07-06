#include "hclib_cpp.h"
#include "hclib_cuda.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system", "cuda" };
    hclib::launch(deps, 2, [] {
        std::cout << "Hello world" << std::endl;
    });
    return 0;
}
