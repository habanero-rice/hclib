#include "hclib_openshmem-internal.h"

#include "hclib-locality-graph.h"

#include <iostream>

static int nic_locale_id;
static hclib::locale_t *nic = NULL;

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
    shmem_init();

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
}

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_finalize) {
    shmem_finalize();
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

HCLIB_REGISTER_MODULE("openshmem", openshmem_pre_initialize, openshmem_post_initialize, openshmem_finalize)
