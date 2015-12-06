#include "hclib.h"
#include "hcpp.h"

extern "C" {

void hclib_init(int * argc, char ** argv) {
    hcpp::init(argc, argv);
}

void hclib_finalize() {
    hcpp::finalize();
}

void async(asyncFct_t fct_ptr, void * arg, struct ddf_st ** ddf_list,
        struct _phased_t * phased_clause, int property) {
    assert(ddf_list == NULL); // not supported
    assert(phased_clause == NULL); // not supported
    assert(property == 0); // not supported

    hcpp::async([=](){ fct_ptr(arg); });
}

void start_finish() {
    hcpp::start_finish();
}

void end_finish() {
    hcpp::end_finish();
}

}
