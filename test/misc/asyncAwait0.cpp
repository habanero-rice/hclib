#include "hclib_cpp.h"

int main(int argc, char ** argv) {
    hclib::launch(&argc, argv, []() {
        hclib::ddf_t * ddf1 = hclib::ddf_create();
        int * val = new int;
        *val = 100;

        hclib::start_finish();
        hclib::asyncAwait(ddf1, [=]() {
            printf("Running asyncAwait \n");
            int* res = (int*) hclib::ddf_get(ddf1);
            printf("ddf_get = %d\n",*res);
        });  
        hclib::async([=]() {
            printf("Start ddf_put\n");
            hclib::ddf_put(ddf1, val);
            printf("End ddf_put\n");
        });
        hclib::end_finish();
    });
    return 0;
}
