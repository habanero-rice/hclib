#include "hclib_cpp.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] {
        std::cout << "Hello world" << std::endl;
    });
    return 0;
}
