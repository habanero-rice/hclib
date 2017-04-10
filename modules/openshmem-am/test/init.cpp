#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_openshmem-am.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system", "openshmem", "openshmem-am" };
    hclib::launch(deps, 3, [] {
        hclib::async_remote([=] {
            printf("Howdy! on PE %d\n", hclib::shmem_my_pe());

            hclib::async([=] { printf("No way\n"); });
        }, 0);
    });
    return 0;
}
