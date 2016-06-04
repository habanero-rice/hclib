#include "hclib_openshmem-internal.h"

#include "hclib-locality-graph.h"

#include <map>
#include <vector>
#include <iostream>

typedef struct _lock_context_t {
    hclib_future_t *last_lock;
    hclib_promise_t * volatile live;
} lock_context_t;

static int nic_locale_id;
static hclib::locale_t *nic = NULL;
static std::map<long *, lock_context_t *> lock_info;
static pthread_mutex_t lock_info_mutex = PTHREAD_MUTEX_INITIALIZER;

static std::vector<hclib::wait_set_t *> waiting_on;
static pthread_mutex_t waiting_on_mutex = PTHREAD_MUTEX_INITIALIZER;

static int pe_to_locale_id(int pe) {
    HASSERT(pe >= 0);
    return -1 * pe - 1;
}

static int locale_id_to_pe(int locale_id) {
    HASSERT(locale_id < 0);
    return (locale_id + 1) * -1;
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_post_initialize) {
    ::shmem_init();

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_finalize) {
    ::shmem_finalize();
}

static hclib::locale_t *get_locale_for_pe(int pe) {
    char name_buf[256];
    sprintf(name_buf, "openshmem-pe-%d", pe);

    hclib::locale_t *new_locale = (hclib::locale_t *)malloc(
            sizeof(hclib::locale_t));
    new_locale->id = pe_to_locale_id(pe);
    new_locale->type = nic_locale_id;
    new_locale->lbl = (char *)malloc(strlen(name_buf) + 1);
    memcpy((void *)new_locale->lbl, name_buf, strlen(name_buf) + 1);
    new_locale->metadata = NULL;
    new_locale->deques = NULL;
    return new_locale;
}

hclib::locale_t *hclib::shmem_my_pe() {
    return get_locale_for_pe(::shmem_my_pe());
}

int hclib::shmem_n_pes() {
    return ::shmem_n_pes();
}

void *hclib::shmem_malloc(size_t size) {
    hclib::promise_t *promise = new hclib::promise_t();
    hclib::async_at(nic, [size, promise] {
        void *alloc = ::shmem_malloc(size);
        promise->put(alloc);
    });

    promise->get_future()->wait();
    void *allocated = promise->get_future()->get();
    delete promise;

    return allocated;
}

void hclib::shmem_free(void *ptr) {
    hclib::finish([ptr] {
        hclib::async_at(nic, [ptr] {
            ::shmem_free(ptr);
        });
    });
}

void hclib::shmem_barrier_all() {
    hclib::finish([] {
        hclib::async_at(nic, [] {
            ::shmem_barrier_all();
        });
    });
}

void hclib::shmem_put64(void *dest, const void *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_at(nic, [dest, source, nelems, pe] {
            ::shmem_put64(dest, source, nelems, pe);
        });
    });
}

void hclib::shmem_broadcast64(void *dest, const void *source, size_t nelems,
        int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync) {
    hclib::finish([dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, pSync] {
        hclib::async_at(nic, [dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, pSync] {
            ::shmem_broadcast64(dest, source, nelems, PE_root, PE_start,
                logPE_stride, PE_size, pSync);
        });
    });
}

hclib::locale_t *hclib::shmem_remote_pe(int pe) {
    return get_locale_for_pe(pe);
}

int hclib::pe_for_locale(hclib::locale_t *locale) {
    return locale_id_to_pe(locale->id);
}

static void *shmem_set_lock_impl(void *arg) {
    ::shmem_set_lock((long *)arg);
    return NULL;
}

static void *shmem_clear_lock_impl(void *arg) {
    ::shmem_clear_lock((long *)arg);
    return NULL;
}

void hclib::shmem_set_lock(volatile long *lock) {
    int err = pthread_mutex_lock(&lock_info_mutex);
    HASSERT(err == 0);

    hclib_future_t *await = NULL;
    lock_context_t *ctx = NULL;

    hclib_promise_t *promise = hclib_promise_create();

    std::map<long *, lock_context_t *>::iterator found = lock_info.find((long *)lock);
    if (found != lock_info.end()) {
        ctx = found->second;
    } else {
        /*
         * Cannot assert that *lock == 0L here as another node may be locking
         * it. Can only guarantee that no one else on the same node has locked
         * it.
         */
        ctx = (lock_context_t *)malloc(sizeof(lock_context_t));
        memset(ctx, 0x00, sizeof(lock_context_t));

        lock_info.insert(std::pair<long *, lock_context_t *>((long *)lock, ctx));
    }

    await = hclib_async_future(shmem_set_lock_impl, (void *)lock, ctx->last_lock, nic);
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
    HASSERT(found != lock_info.end()) // Doesn't make much sense to clear a lock that hasn't been set

    hclib_future_t *await = hclib_async_future(shmem_clear_lock_impl, lock,
            NULL, nic);

    err = pthread_mutex_unlock(&lock_info_mutex);
    HASSERT(err == 0);

    hclib_future_wait(await);

    HASSERT(found->second->live);
    hclib_promise_t *live = found->second->live;
    found->second->live = NULL;
    hclib_promise_put(live, NULL);
}

void hclib::shmem_int_get(int *dest, const int *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_at(nic, [dest, source, nelems, pe] {
            ::shmem_int_get(dest, source, nelems, pe);
        });
    });
}

void hclib::shmem_getmem(void *dest, const void *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_at(nic, [dest, source, nelems, pe] {
            ::shmem_getmem(dest, source, nelems, pe);
        });
    });
}

void hclib::shmem_int_put(int *dest, const int *source, size_t nelems, int pe) {
    hclib::finish([dest, source, nelems, pe] {
        hclib::async_at(nic, [dest, source, nelems, pe] {
            ::shmem_int_put(dest, source, nelems, pe);
        });
    });
}

void hclib::shmem_int_add(int *dest, int value, int pe) {
    hclib::finish([dest, value, pe] {
        hclib::async_at(nic, [dest, value, pe] {
            ::shmem_int_add(dest, value, pe);
        });
    });
}

long long hclib::shmem_longlong_fadd(long long *target, long long value,
        int pe) {
    long long *val_ptr = (long long *)malloc(sizeof(long long));
    hclib::promise_t *promise = new hclib::promise_t();

    hclib::async_at(nic, [target, value, pe, promise, val_ptr] {
        const long long val = ::shmem_longlong_fadd(target, value, pe);
        *val_ptr = val;
        promise->put(NULL);
    });

    promise->get_future()->wait();
    const long long result = *val_ptr;

    delete promise;
    free(val_ptr);
    return result;
}

int hclib::shmem_int_fadd(int *dest, int value, int pe) {
    hclib::promise_t *promise = new hclib::promise_t();
    hclib::async_at(nic, [dest, value, pe, promise] {
        const int fetched = ::shmem_int_fadd(dest, value, pe);
        int *heap_fetched = (int *)malloc(sizeof(int));
        *heap_fetched = fetched;
        promise->put((void *)heap_fetched);
    });

    promise->get_future()->wait();
    int *heap_fetched = (int *)promise->get_future()->get();
    const int fetched = *heap_fetched;
    delete promise;
    free(heap_fetched);

    return fetched;
}

void hclib::shmem_int_sum_to_all(int *target, int *source, int nreduce,
                          int PE_start, int logPE_stride,
                          int PE_size, int *pWrk, long *pSync) {
    hclib::finish([target, source, nreduce, PE_start, logPE_stride, PE_size,
            pWrk, pSync] {
        hclib::async_at(nic, [target, source, nreduce, PE_start, logPE_stride,
            PE_size, pWrk, pSync] {
            ::shmem_int_sum_to_all(target, source, nreduce, PE_start,
                logPE_stride, PE_size, pWrk, pSync);
        });
    });
}

void hclib::shmem_longlong_sum_to_all(long long *target, long long *source,
                               int nreduce, int PE_start,
                               int logPE_stride, int PE_size,
                               long long *pWrk, long *pSync) {
    hclib::finish([target, source, nreduce, PE_start, logPE_stride, PE_size,
            pWrk, pSync] {
        hclib::async_at(nic, [target, source, nreduce, PE_start, logPE_stride,
            PE_size, pWrk, pSync] {
            ::shmem_longlong_sum_to_all(target, source, nreduce, PE_start,
                logPE_stride, PE_size, pWrk, pSync);
        });
    });
}

void hclib::shmem_longlong_p(long long *addr, long long value, int pe) {
    hclib::finish([addr, value, pe] {
        hclib::async_at(nic, [addr, value, pe] {
            ::shmem_longlong_p(addr, value, pe);
        });
    });
}

void hclib::shmem_longlong_put(long long *dest, const long long *src,
                        size_t nelems, int pe) {
    hclib::finish([dest, src, nelems, pe] {
        hclib::async_at(nic, [dest, src, nelems, pe] {
            ::shmem_longlong_put(dest, src, nelems, pe);
        });
    });
}

std::string hclib::shmem_name() {
    std::stringstream ss;
    ss << SHMEM_VENDOR_STRING << " v" << SHMEM_MAJOR_VERSION << "." <<
        SHMEM_MINOR_VERSION << std::endl;
    return ss.str();
}

static void poll_on_waits() {
    int err = pthread_mutex_lock(&waiting_on_mutex);
    HASSERT(err == 0);

    HASSERT(waiting_on.size() > 0);

    std::vector<hclib::wait_set_t *>::iterator curr = waiting_on.begin();
    while (curr != waiting_on.end()) {
        hclib::wait_set_t *wait_set = *curr;

        bool any_complete = false;
        for (int i = 0; i < wait_set->ninfos && !any_complete; i++) {
            hclib::wait_info_t *wait_info = wait_set->infos + i;

            switch (wait_info->cmp) {
                case SHMEM_CMP_EQ:
                    switch (wait_info->type) {
                        case hclib::integer:
                            if (*((volatile int *)wait_info->var) == wait_info->cmp_value.i) {
                                any_complete = true;
                            }
                            break; // integer

                        default:
                            std::cerr << "Unsupported wait type " << wait_info->type << std::endl;
                            exit(1);
                    }
                    break; // SHMEM_CMP_EQ
           
                default:
                    std::cerr << "Unsupported cmp type " << wait_info->cmp << std::endl;
                    exit(1);
            }
        }

        if (any_complete) {
            waiting_on.erase(curr);
            if (wait_set->task) {
                HASSERT(wait_set->signal == NULL);
                spawn(wait_set->task);
            } else {
                HASSERT(wait_set->task == NULL);
                hclib_promise_put(wait_set->signal, NULL);
            }
            free(wait_set->infos);
            free(wait_set);
        } else {
            curr++;
        }
    }

    if (!waiting_on.empty()) {
        hclib::async_at(nic, [] {
            poll_on_waits();
        });
    }

    err = pthread_mutex_unlock(&waiting_on_mutex);
    HASSERT(err == 0);
}

void hclib::shmem_int_wait_until(volatile int *ivar, int cmp, int cmp_value) {
    hclib_promise_t *promise = construct_and_insert_wait_set(&ivar, cmp,
            &cmp_value, 1, integer, i, NULL);
    HASSERT(promise);

    hclib_future_wait(hclib_get_future_for_promise(promise));

    hclib_promise_free(promise);
}

void hclib::shmem_int_wait_until_any(volatile int **ivars, int cmp,
        int *cmp_values, int nwaits) {
    hclib_promise_t *promise = construct_and_insert_wait_set(ivars, cmp,
            cmp_values, nwaits, integer, i, NULL);
    HASSERT(promise);

    hclib_future_wait(hclib_get_future_for_promise(promise));

    hclib_promise_free(promise);
}

void hclib::enqueue_wait_set(hclib::wait_set_t *wait_set) {
    int err = pthread_mutex_lock(&waiting_on_mutex);
    HASSERT(err == 0);

    waiting_on.push_back(wait_set);

    if (waiting_on.size() == 1) {
        hclib::async_at(nic, [] {
            poll_on_waits();
        });
    }

    err = pthread_mutex_unlock(&waiting_on_mutex);
    HASSERT(err == 0);
}

HCLIB_REGISTER_MODULE("openshmem", openshmem_pre_initialize, openshmem_post_initialize, openshmem_finalize)
