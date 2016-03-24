#include "hclib_cpp.h"
#include "hclib_cuda.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        std::cout << "Hello world" << std::endl;
    });
    return 0;
}
