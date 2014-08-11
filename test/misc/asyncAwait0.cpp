#include "hcpp.h"
using namespace hcpp;

int main(int argc, char ** argv) {
    DDF_t** ddf_list = DDF_LIST(2);
    ddf_list[1] = NULL;
    ddf_list[0] = DDF_CREATE();
    int * val = new int;
    *val = 100;

    hcpp::finish([=]() {
      async([&]() {
        printf("Putting in DDF by async\n");
        DDF_PUT(ddf_list[0], val);
      });
      asyncAwait(ddf_list[0], [=]() {
        printf("Running asyncAwait \n");
        int* res = (int*) DDF_GET(ddf_list[0]);
        printf("DDF_GET = %d\n",*res);
      });  
    });
    return 0;
}
