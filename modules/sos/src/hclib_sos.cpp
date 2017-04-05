#include "hclib_sos-internal.h"
#include "hclib-locality-graph.h"

extern "C" {
#include "shmemx.h"
}

#include <map>
#include <vector>
#include <iostream>
#include <sstream>

#ifdef HCLIB_INSTRUMENT
enum SOS_FUNC_LABELS {
    shmem_malloc_lbl = 0,
    shmem_free_lbl,
    shmem_barrier_all_lbl,
    shmem_fence_lbl,
    shmem_quiet_lbl,
    shmem_put64_lbl,
    shmem_broadcast64_lbl,
    shmem_set_lock_lbl,
    shmem_clear_lock_lbl,
    shmem_int_get_lbl,
    shmem_getmem_lbl,
    shmem_putmem_lbl,
    shmem_int_put_lbl,
    shmem_char_put_nbi_lbl,
    shmem_char_put_signal_nbi_lbl,
    shmem_int_add_lbl,
    shmem_longlong_fadd_lbl,
    shmem_int_fadd_lbl,
    shmem_int_sum_to_all_lbl,
    shmem_longlong_sum_to_all_lbl,
    shmem_longlong_max_to_all_lbl,
    shmem_longlong_p_lbl,
    shmem_longlong_put_lbl,
    shmem_int_finc_lbl,
    shmem_int_fetch_lbl,
    shmem_collect32_lbl,
    shmem_fcollect64_lbl,
    shmem_async_when_polling_lbl,
    shmem_int_wait_until_lbl,
    N_SOS_FUNCS
};

const char *SOS_FUNC_NAMES[N_SOS_FUNCS] = {
    "shmem_malloc",
    "shmem_free",
    "shmem_barrier_all",
    "shmem_fence",
    "shmem_quiet",
    "shmem_put64",
    "shmem_broadcast64",
    "shmem_set_lock",
    "shmem_clear_lock",
    "shmem_int_get",
    "shmem_getmem",
    "shmem_putmem",
    "shmem_int_put",
    "shmem_char_put_nbi",
    "shmem_char_put_signal_nbi",
    "shmem_int_add",
    "shmem_longlong_fadd",
    "shmem_int_fadd",
    "shmem_int_sum_to_all",
    "shmem_longlong_sum_to_all",
    "shmem_longlong_max_to_all",
    "shmem_longlong_p",
    "shmem_longlong_put",
    "shmem_int_finc",
    "shmem_int_fetch",
    "shmem_collect32",
    "shmem_fcollect64",
    "shmem_async_when_polling",
    "shmem_int_wait_until"};

static int event_ids[N_SOS_FUNCS];

#define SOS_START_OP(funcname) \
    const unsigned _event_id = hclib_register_event(event_ids[funcname##_lbl], \
            START, -1)
#define SOS_END_OP(funcname) \
    hclib_register_event(event_ids[funcname##_lbl], END, _event_id)

#else
#define SOS_START_OP(funcname)
#define SOS_END_OP(funcname)
#endif

#define SOS_HANG_WORKAROUND

static unsigned domain_ctx_id = 0;
static shmemx_domain_t *domains = NULL;
static shmemx_ctx_t *contexts = NULL;
static int nthreads = -1;

typedef struct _lock_context_t {
    // A future satisfied by the last attempt to lock this global lock.
    hclib_future_t *last_lock;
    /*
     * Store the promise that should be satisfied by whoever is currently in the
     * critical section once they complete it.
     */
    hclib_promise_t * volatile live;
} lock_context_t;

pending_sos_op *pending = NULL;

static int nic_locale_id;
hclib::locale_t *nic = NULL;
static std::map<long *, lock_context_t *> lock_info;
static pthread_mutex_t lock_info_mutex = PTHREAD_MUTEX_INITIALIZER;

bool test_sos_completion(void *generic_op) {
    pending_sos_op *op = (pending_sos_op *)generic_op;

    switch (op->cmp) {
        case SHMEM_CMP_EQ:
            switch (op->type) {
                case integer:
                    if (*((volatile int *)op->var) == op->cmp_value.i) {
                        return true;
                    }
                    break; // integer

                default:
                    std::cerr << "Unsupported wait type " << op->type <<
                        std::endl;
                    exit(1);
            }
        break; // SHMEM_CMP_EQ
              
        case SHMEM_CMP_NE:
            switch (op->type) {
                case integer:
                    if (*((volatile int *)op->var) != op->cmp_value.i) {
                        return true;
                    }
                    break; // integer

                default:
                    std::cerr << "Unsupported wait type " << op->type <<
                        std::endl;
                    exit(1);
            }
            break; // SHMEM_CMP_NE

            default:
                std::cerr << "Unsupported cmp type " << op->cmp << std::endl;
                exit(1);
    }

    return false;
}

HCLIB_MODULE_PRE_INITIALIZATION_FUNC(sos_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
#ifdef HCLIB_INSTRUMENT
    int i;
    for (i = 0; i < N_SOS_FUNCS; i++) {
        event_ids[i] = register_event_type((char *)SOS_FUNC_NAMES[i]);
    }
#endif

}

static void init_sos_state(void *state, void *user_data, int tid) {
    assert(user_data == NULL);
    shmemx_domain_t *domain = (shmemx_domain_t *)state;
    shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    *domain = domains[tid];
    *ctx = contexts[tid];
}

static void release_sos_state(void *state, void *user_data) {
    assert(user_data == NULL);
    shmemx_domain_t *domain = (shmemx_domain_t *)state;
    shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    ::shmemx_ctx_quiet(*ctx);
    ::shmemx_ctx_destroy(*ctx);
    ::shmemx_domain_destroy(1, domain);
}

HCLIB_MODULE_INITIALIZATION_FUNC(sos_post_initialize) {
    int provided_thread_safety;
    const int desired_thread_safety = SHMEMX_THREAD_MULTIPLE;
    ::shmemx_init_thread(desired_thread_safety, &provided_thread_safety);
    assert(provided_thread_safety == desired_thread_safety);

    const int pe = ::shmem_my_pe();
    const int npes = ::shmem_n_pes();

    nthreads = hclib_get_num_workers();
    domains = (shmemx_domain_t *)malloc(hclib_get_num_workers() * sizeof(*domains));
    assert(domains);
    contexts = (shmemx_ctx_t *)malloc(hclib_get_num_workers() * sizeof(*contexts));
    assert(contexts);

    int err = ::shmemx_domain_create(SHMEMX_THREAD_MULTIPLE,
            hclib_get_num_workers(), domains);
    assert(err == 0); 

    for (int i = 0; i < hclib_get_num_workers(); i++) {
        err = ::shmemx_ctx_create(domains[i], contexts + i);
        assert(err == 0);
    }

#ifdef SOS_HANG_WORKAROUND
    const unsigned long long start_time = hclib_current_time_ns();

    int *buf = (int *)::shmem_malloc(npes * sizeof(int));
    assert(buf);
    buf[pe] = pe;

    int i, j;
    for (i = 0; i < npes; i++) {
        for (j = 0; j < hclib_get_num_workers(); j++) {
            const int unused = ::shmemx_ctx_int_fadd(buf, 1, i, contexts[j]);
        }
    }

    ::shmem_barrier_all();
    ::shmem_free(buf);

    const unsigned long long elapsed = hclib_current_time_ns() - start_time;
    if (pe == 0) {
        fprintf(stderr, "SoS hang workaround took %f ms\n",
                (double)elapsed / 1000000.0);
    }
#endif

    domain_ctx_id = hclib_add_per_worker_module_state(
            sizeof(shmemx_domain_t) + sizeof(shmemx_ctx_t), init_sos_state,
            NULL);

    /*
     * This is only still needed because not all OpenSHMEM APIs are
     * contexts-based.
     */
    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
}

HCLIB_MODULE_INITIALIZATION_FUNC(sos_finalize) {
    hclib_release_per_worker_module_state(domain_ctx_id, release_sos_state,
            NULL);
    ::shmem_finalize();
}

int hclib::shmem_my_pe() {
    return ::shmem_my_pe();
}

int hclib::shmem_n_pes() {
    return ::shmem_n_pes();
}

void *hclib::shmem_malloc(size_t size) {
    void **out_alloc = (void **)malloc(sizeof(void *));
    hclib::finish([out_alloc, size] {
        hclib::async_nb_at([size, out_alloc] {
            SOS_START_OP(shmem_malloc);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_malloc: Allocating " << size <<
                    " bytes" << std::endl;
#endif
            *out_alloc = ::shmem_malloc(size);
            SOS_END_OP(shmem_malloc);
        }, nic);
    });

    void *allocated = *out_alloc;
    free(out_alloc);

    return allocated;
}

void hclib::shmem_free(void *ptr) {
    hclib::finish([ptr] {
        hclib::async_nb_at([ptr] {
            SOS_START_OP(shmem_free);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_free: ptr=" << ptr <<
                    std::endl;
#endif
            ::shmem_free(ptr);
            SOS_END_OP(shmem_free);
        }, nic);
    });
}

void hclib::shmem_barrier_all() {
    for (int i = 0; i < nthreads; i++) {
        ::shmemx_ctx_quiet(contexts[i]);
    }

    hclib::finish([] {
        hclib::async_nb_at([] {
            SOS_START_OP(shmem_barrier_all);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_barrier_all" << std::endl;
#endif
            ::shmem_barrier_all();
            SOS_END_OP(shmem_barrier_all);
        }, nic);
    });
}

void hclib::shmem_fence() {
    hclib::finish([] {
        hclib::async_nb_at([] {
            SOS_START_OP(shmem_fence);
            ::shmem_fence();
            SOS_END_OP(shmem_fence);
        }, nic);
    });
}

void hclib::shmem_quiet() {
    hclib::finish([] {
        hclib::async_nb_at([] {
            SOS_START_OP(shmem_quiet);
            ::shmem_quiet();
            SOS_END_OP(shmem_quiet);
        }, nic);
    });
}

void hclib::shmem_put64(void *dest, const void *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_nb_at([dest, source, nelems, pe] {
            SOS_START_OP(shmem_put64);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_put64: dest=" << dest <<
                    " source=" << source << " nelems=" << nelems << " pe=" <<
                    pe << std::endl;
#endif
            ::shmem_put64(dest, source, nelems, pe);
            SOS_END_OP(shmem_put64);
        }, nic);
    });
}

void hclib::shmem_broadcast64(void *dest, const void *source, size_t nelems,
        int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync) {
    hclib::finish([dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, pSync] {
        hclib::async_nb_at([dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, pSync] {
            SOS_START_OP(shmem_broadcast64);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_broadcast64: dest=" <<
                    dest << " source=" << source << " nelems=" << nelems <<
                    " PE_root=" << PE_root << " PE_start=" << PE_start <<
                    " logPE_stride=" << logPE_stride << " PE_size=" <<
                    PE_size << std::endl;
#endif
            ::shmem_broadcast64(dest, source, nelems, PE_root, PE_start,
                logPE_stride, PE_size, pSync);
            SOS_END_OP(shmem_broadcast64);
        }, nic);
    });
}

static void *shmem_set_lock_impl(void *arg) {
    SOS_START_OP(shmem_set_lock);
#ifdef TRACE
    std::cerr << ::shmem_my_pe() << ": shmem_set_lock: lock=" << arg <<
        std::endl;
#endif
    ::shmem_set_lock((long *)arg);
    SOS_END_OP(shmem_set_lock);
    return NULL;
}

static void shmem_clear_lock_impl(void *arg) {
    SOS_START_OP(shmem_clear_lock);
#ifdef TRACE
    std::cerr << ::shmem_my_pe() << ": shmem_clear_lock: lock=" << arg <<
        std::endl;
#endif
    ::shmem_clear_lock((long *)arg);
    SOS_END_OP(shmem_clear_lock);
}

void hclib::shmem_set_lock(volatile long *lock) {
    int err = pthread_mutex_lock(&lock_info_mutex);
    HASSERT(err == 0);

    hclib_future_t *await = NULL;
    lock_context_t *ctx = NULL;

    hclib_promise_t *promise = hclib_promise_create();

    std::map<long *, lock_context_t *>::iterator found =
        lock_info.find((long *)lock);
    if (found != lock_info.end()) {
        ctx = found->second;
    } else {
        ctx = (lock_context_t *)calloc(1, sizeof(lock_context_t));

        lock_info.insert(std::pair<long *, lock_context_t *>((long *)lock, ctx));
    }

    /*
     * Launch an async at the NIC that performs the actual lock. This task's
     * execution is predicated on the last lock, which may be NULL.
     */
    await = hclib_async_future(shmem_set_lock_impl, (void *)lock,
            &ctx->last_lock, 1, nic);
    // Save ourselves as the last person to lock.
    ctx->last_lock = hclib_get_future_for_promise(promise);

    err = pthread_mutex_unlock(&lock_info_mutex);
    HASSERT(err == 0);

    hclib_future_wait(await);
    HASSERT(ctx->live == NULL);
    ctx->live = promise;
}

void hclib::shmem_clear_lock(long *lock) {
    int err = pthread_mutex_lock(&lock_info_mutex);
    HASSERT(err == 0);

    std::map<long *, lock_context_t *>::iterator found = lock_info.find(lock);
    // Doesn't make much sense to clear a lock that hasn't been set
    HASSERT(found != lock_info.end())

    err = pthread_mutex_unlock(&lock_info_mutex);
    HASSERT(err == 0);

    hclib::finish([&] {
        hclib_async_nb(shmem_clear_lock_impl, lock, nic);
    });

    HASSERT(found->second->live);
    hclib_promise_t *live = found->second->live;
    found->second->live = NULL;
    hclib_promise_put(live, NULL);
}

void hclib::shmem_int_get(int *dest, const int *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_nb_at([dest, source, nelems, pe] {
            SOS_START_OP(shmem_int_get);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_int_get: dest=" << dest <<
                    " source=" << source << " nelems=" << nelems << " pe=" <<
                    pe << std::endl;
#endif
            ::shmem_int_get(dest, source, nelems, pe);
            SOS_END_OP(shmem_int_get);
        }, nic);
    });
}

void hclib::shmem_getmem(void *dest, const void *source, size_t nelems, int pe) {
    void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    assert(state);
    shmemx_domain_t *domain = (shmemx_domain_t *)state;
    shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    shmemx_ctx_getmem(dest, source, nelems, pe, *ctx);
}

void hclib::shmem_putmem(void *dest, const void *source, size_t nelems, int pe) {
    void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    assert(state);
    shmemx_domain_t *domain = (shmemx_domain_t *)state;
    shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    shmemx_ctx_putmem(dest, source, nelems, pe, *ctx);
}

void hclib::shmem_int_put(int *dest, const int *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_nb_at([dest, source, nelems, pe] {
            SOS_START_OP(shmem_int_put);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_int_put: dest=" << dest <<
                    " source=" << source << " nelems=" << nelems << " pe=" <<
                    pe << std::endl;
#endif
            ::shmem_int_put(dest, source, nelems, pe);
            SOS_END_OP(shmem_int_put);
        }, nic);
    });
}

void hclib::shmem_char_put_nbi(char *dest, const char *source, size_t nelems,
        int pe) {
    hclib::finish([&] {
        hclib::async_nb_at([&] {
            SOS_START_OP(shmem_char_put_nbi);
            ::shmem_char_put_nbi(dest, source, nelems, pe);
            SOS_END_OP(shmem_char_put_nbi);
        }, nic);
    });
}

void hclib::shmem_char_put_signal_nbi(char *dest, const char *source,
        size_t nelems, char *signal_dest, const char *signal_source,
        size_t signal_nelems, int pe) {
    hclib::finish([&] {
        hclib::async_nb_at([&] {
            SOS_START_OP(shmem_char_put_signal_nbi);
            ::shmem_char_put_nbi(dest, source, nelems, pe);
            ::shmem_fence();
            ::shmem_char_put_nbi(signal_dest, signal_source, signal_nelems, pe);
            SOS_END_OP(shmem_char_put_signal_nbi);
        }, nic);
    });
}

void hclib::shmem_int_add(int *dest, int value, int pe) {
    hclib::finish([dest, value, pe] {
        hclib::async_nb_at([dest, value, pe] {
            SOS_START_OP(shmem_int_add);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_int_add: dest=" << dest <<
                    " value=" << value << " pe=" << pe << std::endl;
#endif
            ::shmem_int_add(dest, value, pe);
            SOS_END_OP(shmem_int_add);
        }, nic);
    });
}

long long hclib::shmem_longlong_fadd(long long *target, long long value,
        int pe) {
    void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    assert(state);
    shmemx_domain_t *domain = (shmemx_domain_t *)state;
    shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    return shmemx_ctx_longlong_fadd(target, value, pe, *ctx);

//     long long *val_ptr = (long long *)malloc(sizeof(long long));
//     hclib::finish([target, value, pe, val_ptr] {
//         hclib::async_nb_at([target, value, pe, val_ptr] {
//             SOS_START_OP(shmem_longlong_fadd);
// #ifdef TRACE
//             std::cerr << ::shmem_my_pe() << ": shmem_longlong_fadd: target=" <<
//                 target << " value=" << value << " pe=" << pe << std::endl;
// #endif
//             const long long val = ::shmem_longlong_fadd(target, value, pe);
//             *val_ptr = val;
//             SOS_END_OP(shmem_longlong_fadd);
//         }, nic);
//     });
// 
//     const long long result = *val_ptr;
// 
//     free(val_ptr);
//     return result;
}

int hclib::shmem_int_fadd(int *dest, int value, int pe) {
    int *heap_fetched = (int *)malloc(sizeof(int));
    hclib::finish([dest, value, pe, heap_fetched] {
        hclib::async_nb_at([dest, value, pe, heap_fetched] {
            SOS_START_OP(shmem_int_fadd);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_int_fadd: dest=" <<
                dest << " value=" << value << " pe=" << pe << std::endl;
#endif
            const int fetched = ::shmem_int_fadd(dest, value, pe);
            *heap_fetched = fetched;
            SOS_END_OP(shmem_int_fadd);
        }, nic);
    });

    const int fetched = *heap_fetched;
    free(heap_fetched);

    return fetched;
}

int hclib::shmem_int_swap(int *dest, int value, int pe) {
    int *heap_fetched = (int *)malloc(sizeof(int));
    hclib::finish([dest, value, pe, heap_fetched] {
        hclib::async_nb_at([dest, value, pe, heap_fetched] {
            const int fetched = ::shmem_int_swap(dest, value, pe);
            *heap_fetched = fetched;
        }, nic);
    });

    const int fetched = *heap_fetched;
    free(heap_fetched);

    return fetched;

    // void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    // assert(state);
    // shmemx_domain_t *domain = (shmemx_domain_t *)state;
    // shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    // return shmemx_ctx_int_swap(dest, value, pe, *ctx);
}

int hclib::shmem_int_cswap(int *dest, int cond, int value, int pe) {
    int *heap_fetched = (int *)malloc(sizeof(int));
    hclib::finish([dest, cond, value, pe, heap_fetched] {
        hclib::async_nb_at([dest, cond, value, pe, heap_fetched] {
            const int fetched = ::shmem_int_cswap(dest, cond, value, pe);
            *heap_fetched = fetched;
        }, nic);
    });

    const int fetched = *heap_fetched;
    free(heap_fetched);

    return fetched;

    // void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    // assert(state);
    // shmemx_domain_t *domain = (shmemx_domain_t *)state;
    // shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    // return shmemx_ctx_int_cswap(dest, cond, value, pe, *ctx);
}

long hclib::shmem_long_finc(long *dest, int pe) {
    long *heap_fetched = (long *)malloc(sizeof(long));
    hclib::finish([dest, pe, heap_fetched] {
        hclib::async_nb_at([dest, pe, heap_fetched] {
            const long fetched = ::shmem_long_finc(dest,pe);
            *heap_fetched = fetched;
        }, nic);
    });

    const long fetched = *heap_fetched;
    free(heap_fetched);

    return fetched;

    // void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    // assert(state);
    // shmemx_domain_t *domain = (shmemx_domain_t *)state;
    // shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    // return shmemx_ctx_long_finc(dest, pe, *ctx);
}

int hclib::shmem_int_finc(int *dest, int pe) {
    int *heap_fetched = (int *)malloc(sizeof(int));
    hclib::finish([dest, pe, heap_fetched] {
        hclib::async_nb_at([dest, pe, heap_fetched] {
            const int fetched = ::shmem_int_finc(dest,pe);
            *heap_fetched = fetched;
        }, nic);
    });

    const int fetched = *heap_fetched;
    free(heap_fetched);

    return fetched;

    // void *state = hclib_get_curr_worker_module_state(domain_ctx_id);
    // assert(state);
    // shmemx_domain_t *domain = (shmemx_domain_t *)state;
    // shmemx_ctx_t *ctx = (shmemx_ctx_t *)(domain + 1);

    // return shmemx_ctx_int_finc(dest, pe, *ctx);
}

void hclib::shmem_int_sum_to_all(int *target, int *source, int nreduce,
                          int PE_start, int logPE_stride,
                          int PE_size, int *pWrk, long *pSync) {
    hclib::finish([target, source, nreduce, PE_start, logPE_stride, PE_size,
            pWrk, pSync] {
        hclib::async_nb_at([target, source, nreduce, PE_start, logPE_stride,
            PE_size, pWrk, pSync] {
            SOS_START_OP(shmem_int_sum_to_all);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_int_sum_to_all: target=" <<
                target << " source=" << source << " nreduce=" << nreduce <<
                " PE_start=" << PE_start << " logPE_stride=" << logPE_stride <<
                " PE_size=" << PE_size << std::endl;
#endif
            ::shmem_int_sum_to_all(target, source, nreduce, PE_start,
                logPE_stride, PE_size, pWrk, pSync);
            SOS_END_OP(shmem_int_sum_to_all);
        }, nic);
    });
}

void hclib::shmem_longlong_sum_to_all(long long *target, long long *source,
                               int nreduce, int PE_start,
                               int logPE_stride, int PE_size,
                               long long *pWrk, long *pSync) {
    hclib::finish([target, source, nreduce, PE_start, logPE_stride, PE_size,
            pWrk, pSync] {
        hclib::async_nb_at([target, source, nreduce, PE_start, logPE_stride,
            PE_size, pWrk, pSync] {
            SOS_START_OP(shmem_longlong_sum_to_all);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_longlong_sum_to_all: "
                "target=" << target << " source=" << source << " nreduce=" <<
                nreduce << " PE_start=" << PE_start << " logPE_stride=" <<
                logPE_stride << " PE_size=" << PE_size << std::endl;
#endif
            ::shmem_longlong_sum_to_all(target, source, nreduce, PE_start,
                logPE_stride, PE_size, pWrk, pSync);
            SOS_END_OP(shmem_longlong_sum_to_all);
        }, nic);
    });
}

void hclib::shmem_longlong_max_to_all(long long *target, long long *source,
                               int nreduce, int PE_start,
                               int logPE_stride, int PE_size,
                               long long *pWrk, long *pSync) {
    hclib::finish([&] {
        hclib::async_nb_at([&] {
            SOS_START_OP(shmem_longlong_max_to_all);
            ::shmem_longlong_max_to_all(target, source, nreduce, PE_start,
                logPE_stride, PE_size, pWrk, pSync);
            SOS_END_OP(shmem_longlong_max_to_all);
        }, nic);
    });
}

void hclib::shmem_longlong_p(long long *addr, long long value, int pe) {
    hclib::finish([addr, value, pe] {
        hclib::async_nb_at([addr, value, pe] {
            SOS_START_OP(shmem_longlong_p);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_longlong_p: addr=" <<
                addr << " value=" << value << " pe=" << pe << std::endl;
#endif
            ::shmem_longlong_p(addr, value, pe);
            SOS_END_OP(shmem_longlong_p);
        }, nic);
    });
}

void hclib::shmem_longlong_put(long long *dest, const long long *src,
                        size_t nelems, int pe) {
    hclib::finish([dest, src, nelems, pe] {
        hclib::async_nb_at([dest, src, nelems, pe] {
            SOS_START_OP(shmem_longlong_put);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_longlong_put: dest=" <<
                dest << " src=" << src << "nelems=" << nelems << " pe=" << pe <<
                std::endl;
#endif
            ::shmem_longlong_put(dest, src, nelems, pe);
            SOS_END_OP(shmem_longlong_put);
        }, nic);
    });
}

void hclib::shmem_collect32(void *dest, const void *source, size_t nelems,
        int PE_start, int logPE_stride, int PE_size, long *pSync) {
    hclib::finish([dest, source, nelems, PE_start, logPE_stride, PE_size, pSync] {
        hclib::async_nb_at([dest, source, nelems, PE_start, logPE_stride, PE_size, pSync] {
            SOS_START_OP(shmem_collect32);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_collect32" << std::endl;
#endif
            ::shmem_collect32(dest, source, nelems, PE_start, logPE_stride,
                PE_size, pSync);
            SOS_END_OP(shmem_collect32);
        }, nic);
    });
}

void hclib::shmem_fcollect64(void *dest, const void *source, size_t nelems,
        int PE_start, int logPE_stride, int PE_size, long *pSync) {
    hclib::finish([&] {
        hclib::async_nb_at([&] {
            SOS_START_OP(shmem_fcollect64);
#ifdef TRACE
            std::cerr << ::shmem_my_pe() << ": shmem_fcollect64" << std::endl;
#endif
            ::shmem_fcollect64(dest, source, nelems, PE_start, logPE_stride,
                PE_size, pSync);
            SOS_END_OP(shmem_fcollect64);
        }, nic);
    });

}

std::string hclib::shmem_name() {
    std::stringstream ss;
#ifdef SHMEM_VENDOR_STRING
    ss << SHMEM_VENDOR_STRING << " v" << SHMEM_MAJOR_VERSION << "." <<
        SHMEM_MINOR_VERSION << std::endl;
#else
    ss << "Unknown Impl" << std::endl;
#endif
    return ss.str();
}

/*
 * Implement a blocking wait as a wait on a future, which results in a context
 * switch and allows us to not waste a hardware or software thread spinning.
 */
void hclib::shmem_int_wait_until(volatile int *ivar, int cmp, int cmp_value) {
    SOS_START_OP(shmem_int_wait_until);
    hclib::promise_t<void> *prom = new hclib::promise_t<void>();

    pending_sos_op *op = (pending_sos_op *)malloc(sizeof(*op));
    assert(op);

    op->type = integer;
    op->var = ivar;
    op->cmp = cmp;
    op->cmp_value.i = cmp_value;
    op->prom = prom;
    op->task = NULL;
#ifdef HCLIB_INSTRUMENT
    op->event_type = event_ids[shmem_int_wait_until_lbl];
    op->event_id = _event_id;
#endif
    hclib::append_to_pending(op, &pending, test_sos_completion, nic);

    prom->get_future()->wait();

    delete prom;
}

HCLIB_REGISTER_MODULE("sos", sos_pre_initialize, sos_post_initialize, sos_finalize)
