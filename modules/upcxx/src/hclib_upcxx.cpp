 #include "hclib_upcxx.h"

#include "hclib-locality-graph.h"

#include <map>
#include <vector>
#include <iostream>

#if defined(HCLIB_MEASURE_START_LATENCY) || defined(HCLIB_PROFILE)
const char *UPCXX_FUNC_NAMES[N_UPCXX_FUNCS] = {
    "barrier",
    "async_wait",
    "advance",
    "async_after",
    "async_copy",
    "async"
};
#endif

#ifdef HCLIB_MEASURE_START_LATENCY
unsigned long long upcxx_latency_counters[N_UPCXX_FUNCS];
unsigned long long upcxx_latency_times[N_UPCXX_FUNCS];
#endif

#ifdef HCLIB_PROFILE
unsigned long long upcxx_profile_counters[N_UPCXX_FUNCS];
unsigned long long upcxx_profile_times[N_UPCXX_FUNCS];
#endif

static int nic_locale_id;
static hclib::locale_t *nic = NULL;

HCLIB_MODULE_INITIALIZATION_FUNC(upcxx_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");

#ifdef HCLIB_MEASURE_START_LATENCY
    memset(upcxx_latency_counters, 0x00, N_UPCXX_FUNCS * sizeof(unsigned long long));
    memset(upcxx_latency_times, 0x00, N_UPCXX_FUNCS * sizeof(unsigned long long));
#endif

#ifdef HCLIB_PROFILE
    memset(upcxx_profile_counters, 0x00, N_UPCXX_FUNCS * sizeof(unsigned long long));
    memset(upcxx_profile_times, 0x00, N_UPCXX_FUNCS * sizeof(unsigned long long));
#endif
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

std::set<hclib::upcxx::event *> seen_events;

uint32_t ranks() {
    return ::upcxx::ranks();
}

uint32_t myrank() {
    return ::upcxx::myrank();
}

void barrier() {
    UPCXX_START_LATENCY;

    hclib::finish([&] {
        hclib::async_at(nic, [&] {
            UPCXX_END_LATENCY(barrier);
            UPCXX_START_PROFILE;
            ::upcxx::barrier();
            UPCXX_END_PROFILE(barrier);
        });
    });
}

void async_wait() {
    UPCXX_START_LATENCY;

    hclib::finish([&] {
        hclib::async_at(nic, [&] {
            UPCXX_END_LATENCY(async_wait);
            UPCXX_START_PROFILE;
            ::upcxx::async_wait();
            UPCXX_END_PROFILE(async_wait);
        });
    });
}

int advance() {
    UPCXX_START_LATENCY;

    hclib::finish([&] {
        hclib::async_at(nic, [&] {
            UPCXX_END_LATENCY(advance);
            UPCXX_START_PROFILE;
            ::upcxx::advance();
            UPCXX_END_PROFILE(advance);
        });
    });
}

hclib::upcxx::gasnet_launcher<::upcxx::rank_t> async(::upcxx::rank_t rank,
        hclib::upcxx::event *ack) {
    UPCXX_START_LATENCY;

    hclib::upcxx::gasnet_launcher<::upcxx::rank_t> *result = NULL;

    hclib::finish([&] {
        hclib::async_at(nic, [&] {
            UPCXX_END_LATENCY(async);
            UPCXX_START_PROFILE;
            result = new hclib::upcxx::gasnet_launcher<::upcxx::rank_t>(
                ::upcxx::async(rank, ack));
            UPCXX_END_PROFILE(async);
        });
    });
    return *result;
}

bool is_memory_shared_with(::upcxx::rank_t r) {
    return ::upcxx::is_memory_shared_with(r);
}

hclib::locale_t *nic_place() { return nic; }

void print_upcxx_profiling_data() {
#if defined(HCLIB_PROFILE) || defined(HCLIB_MEASURE_START_LATENCY)
    printf("PE %d UPCXX PROFILE INFO:\n", myrank());
    for (int i = 0; i < N_UPCXX_FUNCS; i++) {
#ifdef HCLIB_PROFILE
        if (upcxx_profile_counters[i] > 0) {
            printf("  %s: %llu calls, %llu ms\n", UPCXX_FUNC_NAMES[i],
                    upcxx_profile_counters[i],
                    upcxx_profile_times[i] / 1000000);
        }
#endif
#ifdef HCLIB_MEASURE_START_LATENCY
        if (upcxx_latency_counters[i] > 0) {
            printf("  %s: %llu calls, %llu ns mean launch-to-exec latency\n",
                    UPCXX_FUNC_NAMES[i],
                    upcxx_latency_counters[i],
                    upcxx_latency_times[i] / upcxx_latency_counters[i]);
        }

#endif
    }
#endif
}

}
}
