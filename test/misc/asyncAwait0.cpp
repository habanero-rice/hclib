#include "hclib_cpp.h"

int main(int argc, char ** argv) {
    hclib::launch(&argc, argv, []() {
        hclib::promise_t * promise1 = hclib::promise_create();
        int * val = new int;
        *val = 100;

        hclib::start_finish();
        hclib::asyncAwait(promise1, [=]() {
            printf("Running asyncAwait \n");
            int* res = (int*) hclib::promise_get(promise1);
            printf("promise_get = %d\n",*res);
        });  
        hclib::async([=]() {
            printf("Start promise_put\n");
            hclib::promise_put(promise1, val);
            printf("End promise_put\n");
        });
        hclib::end_finish();
    });
    return 0;
}
