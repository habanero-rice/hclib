 #include "hclib_upcxx.h"

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
    ::upcxx::init(NULL, NULL);

    hclib::upcxx::team_all = ::upcxx::team_all;

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
}

HCLIB_MODULE_INITIALIZATION_FUNC(upcxx_finalize) {
    ::upcxx::finalize();
}

HCLIB_REGISTER_MODULE("upcxx", upcxx_pre_initialize, upcxx_post_initialize, upcxx_finalize)

namespace hclib {

namespace upcxx {

team team_all;

uint32_t ranks() {
    return ::upcxx::ranks();
}

uint32_t myrank() {
    return ::upcxx::myrank();
}

void barrier() {
    hclib::finish([] {
        hclib::async_at(nic, [] {
            ::upcxx::barrier();
        });
    });
}

void async_wait() {
    hclib::finish([] {
        hclib::async_at(nic, [] {
            ::upcxx::async_wait();
        });
    });
}

hclib::upcxx::gasnet_launcher<::upcxx::rank_t> async_after(::upcxx::rank_t rank,
        hclib::upcxx::event *after, hclib::upcxx::event *ack) {
    hclib::upcxx::gasnet_launcher<::upcxx::rank_t> *result = NULL;

    hclib::finish([&rank, &after, &ack, &result] {
        hclib::async_at(nic, [&rank, &after, &ack, &result] {
            result = new hclib::upcxx::gasnet_launcher<::upcxx::rank_t>(
                ::upcxx::async_after(rank, after, ack));
        });
    });
    return *result;
}

bool is_memory_shared_with(::upcxx::rank_t r) {
    return ::upcxx::is_memory_shared_with(r);
}

hclib::locale_t *nic_place() { return nic; }

}
}
