#include "hclib_openshmem-am-internal.h"

#include "hclib-locality-graph.h"

#include <map>
#include <vector>
#include <iostream>
#include <sstream>

// #define TRACE
// #define PROFILE
// #define DETAILED_PROFILING
// #define TRACING

#ifdef PROFILE
static bool disable_profiling = false;

#define START_PROFILE const unsigned long long __start_time = hclib_current_time_ns();

#if defined(TRACING)
static FILE *trace_fp = NULL;
#define END_PROFILE(funcname) { \
    if (!disable_profiling) { \
        fprintf(trace_fp, "TRACE %d : %s : %llu : %llu\n", ::shmem_my_pe(), \
                FUNC_NAMES[funcname##_lbl], __start_time, \
                hclib_current_time_ns()); \
    } \
}
#elif defined(DETAILED_PROFILING)
#define END_PROFILE(funcname) { \
    if (!disable_profiling) { \
        const unsigned long long __end_time = hclib_current_time_ns(); \
        func_counters[funcname##_lbl]++; \
        func_times[funcname##_lbl] += (__end_time - __start_time); \
        printf("%s: %llu ns\n", FUNC_NAMES[funcname##_lbl], \
                (__end_time - __start_time)); \
    } \
}
#else
#define END_PROFILE(funcname) { \
    if (!disable_profiling) { \
        const unsigned long long __end_time = hclib_current_time_ns(); \
        func_counters[funcname##_lbl]++; \
        func_times[funcname##_lbl] += (__end_time - __start_time); \
    } \
}
#endif

enum FUNC_LABELS {
    N_FUNCS
};

const char *FUNC_NAMES[N_FUNCS] = {
};

unsigned long long func_counters[N_FUNCS];
unsigned long long func_times[N_FUNCS];
#else
#define START_PROFILE
#define END_PROFILE(funcname)
#endif

static int nic_locale_id;
static hclib::locale_t *nic = NULL;
int handler_func_id = 420;

void handler_func(void *msg_new, size_t nbytes, int req_pe,
        shmemx_am_token_t token) {
    assert(nbytes >= sizeof(void *));
    void (*fp)(void *) = *((void (**)(void *))msg_new);
    void *payload = ((void (**)(void *))msg_new) + 1;
    fp(payload);
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_am_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
#ifdef PROFILE
    memset(func_counters, 0x00, sizeof(func_counters));
    memset(func_times, 0x00, sizeof(func_times));
#endif
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_am_post_initialize) {
#ifdef PROFILE
#ifdef TRACING
    const char *trace_dir = getenv("HIPER_TRACE_DIR");
    assert(trace_dir);
    char pe_filename[1024];
    sprintf(pe_filename, "%s/%d.trace", trace_dir, ::shmem_my_pe());
    trace_fp = fopen(pe_filename, "w");
    assert(trace_fp);
#endif
#endif

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];

    shmemx_am_attach(handler_func_id, &handler_func);
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_am_finalize) {
#ifdef PROFILE
#ifdef TRACING
    fclose(trace_fp);
#endif
#endif
}

HCLIB_REGISTER_MODULE("openshmem-am", openshmem_am_pre_initialize,
        openshmem_am_post_initialize, openshmem_am_finalize)
