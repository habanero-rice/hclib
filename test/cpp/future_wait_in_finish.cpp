#include <hclib_cpp.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] {
        hclib::finish([] {
            hclib::future_t *await = hclib::async_future([] {
                printf("Hello from future\n");
            });
            await->wait();
        });
    });
    printf("Done\n");
    return 0;
}
