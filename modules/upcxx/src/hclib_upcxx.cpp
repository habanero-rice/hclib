 #include "hclib_upcxx-internal.h"

#include "hclib-locality-graph.h"

#include <map>
#include <vector>
#include <iostream>

static int nic_locale_id;
static hclib::locale_t *nic = NULL;

HCLIB_MODULE_INITIALIZATION_FUNC(upcxx_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
}

HCLIB_MODULE_INITIALIZATION_FUNC(upcxx_post_initialize) {
    upcxx::init(NULL, NULL);

    std::cerr << "UPC++ module initializing on rank " << upcxx::myrank() << std::endl;

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
}

HCLIB_MODULE_INITIALIZATION_FUNC(upcxx_finalize) {
    upcxx::finalize();
}

HCLIB_REGISTER_MODULE("upcxx", upcxx_pre_initialize, upcxx_post_initialize, upcxx_finalize)

namespace hclib {

namespace upcxx {

uint32_t ranks() {
    uint32_t out;
    // hclib::finish([&out] {
    //     hclib::async_at(nic, [&out] {
            out = ::upcxx::ranks();
    //     });
    // });
    return out;
}

uint32_t myrank() {
    uint32_t out;
    // hclib::finish([&out] {
    //     hclib::async_at(nic, [&out] {
            out = ::upcxx::myrank();
    //     });
    // });
    return out;
}

}
}
