#include "hcpp.h"

int main(int argc, char ** argv) {
    hcpp::init(&argc, argv);
    hcpp::ddf_t* ddf1 = hcpp::ddf_create();
    int * val = new int;
    *val = 100;

    hcpp::start_finish();
    hcpp::asyncAwait(ddf1, [=]() {
        printf("Running asyncAwait \n");
        int* res = (int*) hcpp::ddf_get(ddf1);
        printf("ddf_get = %d\n",*res);
    });  
    hcpp::async([=]() {
        printf("Start ddf_put\n");
        hcpp::ddf_put(ddf1, val);
        printf("End ddf_put\n");
    });
    hcpp::end_finish();
    hcpp::finalize();
    return 0;
}
