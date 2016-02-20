#include <hclib_cpp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    hclib::launch(&argc, argv, [=] {
        assert(argc == 1);
    });
    return 0;
}
