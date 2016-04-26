#include <hclib_cpp.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char **argv) {
    hclib::launch([] {
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
