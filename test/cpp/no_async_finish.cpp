#include <hclib_cpp.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
            hclib::finish([] {
                sleep(3);
            });
    });
    std::cout << "Done!" << std::endl;
    return 0;
}
