#include "hclib_atomic.h"

hclib_atomic_t *hclib_atomic_create(const size_t ele_size_in_bytes,
        atomic_init_func init, void *user_data) {
    hclib_atomic_t *atomic = (hclib_atomic_t *)malloc(sizeof(hclib_atomic_t));
    assert(atomic);

    hclib_atomic_init(atomic, ele_size_in_bytes, init, user_data);
    return atomic;
}

void hclib_atomic_init(hclib_atomic_t *atomic, const size_t ele_size,
        atomic_init_func init, void *user_data) {
    int i;

    assert(atomic);
    assert(init);
    assert(ele_size > 0);

    atomic->nthreads = hclib_get_num_workers();
    atomic->val_size = ele_size;
    size_t padded_ele_size = CACHE_LINE_LEN_IN_BYTES -
        (ele_size % CACHE_LINE_LEN_IN_BYTES);
    padded_ele_size += ele_size;
    atomic->padded_val_size = padded_ele_size;
    atomic->vals = (char *)malloc(atomic->nthreads * padded_ele_size);
    assert(atomic->vals);

    for (i = 0; i < atomic->nthreads; i++) {
        init(atomic->vals + i * atomic->padded_val_size, user_data);
    }
}

void hclib_atomic_update(hclib_atomic_t *atomic, atomic_update_func f,
        void *user_data) {
    const int tid = hclib_get_current_worker();
    f(atomic->vals + tid * atomic->padded_val_size, user_data);
}

void *hclib_atomic_gather(hclib_atomic_t *atomic, atomic_gather_func f,
        void *user_data) {
    int i;

    void *result = malloc(atomic->padded_val_size);
    assert(result);

    memcpy(result, atomic->vals, atomic->padded_val_size);

    for (i = 1; i < atomic->nthreads; i++) {
        f(result, atomic->vals + i * atomic->padded_val_size, user_data);
    }

    return result;
}
